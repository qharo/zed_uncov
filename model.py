import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Generator, Tuple, Dict
import config
import utils
import math

class AtrousProbabilityClassifier(nn.Module):
    def __init__(self,
                 in_ch: int,
                 C: int,
                 num_params: int,
                 K: int = 10,
                 kernel_size: int = 3,
                 atrous_rates_str: str = '1,2,4') -> None:
        super(AtrousProbabilityClassifier, self).__init__()

        Kp = num_params * C * K

        self.atrous = StackedAtrousConvs(atrous_rates_str, in_ch, Kp,
                                         kernel_size=kernel_size)
        self._repr = f'C={C}; K={K}; Kp={Kp}; rates={atrous_rates_str}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        :param x: N C H W
        :return: N Kp H W
        """
        return self.atrous(x)


class StackedAtrousConvs(nn.Module):
    def __init__(self,
                 atrous_rates_str: Union[str, int],
                 Cin: int,
                 Cout: int,
                 bias: bool = True,
                 kernel_size: int = 3) -> None:
        super(StackedAtrousConvs, self).__init__()
        atrous_rates = self._parse_atrous_rates_str(atrous_rates_str)
        self.atrous = nn.ModuleList(
            [utils.create_conv_layer(Cin, Cin, kernel_size, rate=rate)
             for rate in atrous_rates])
        self.lin = utils.create_conv_layer(len(atrous_rates) * Cin, Cout, 1, bias=bias)

    @staticmethod
    def _parse_atrous_rates_str(atrous_rates_str: Union[str, int]) -> List[int]:
        if isinstance(atrous_rates_str, int):
            return [atrous_rates_str]
        else:
            return list(map(int, atrous_rates_str.split(',')))


    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = torch.cat([atrous(x)
                       for atrous in self.atrous], dim=1)  # type: ignore
        x = self.lin(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, kernel_size: int, act: str = "leaky_relu", atrous: int = 1, bn: bool = False) -> None:
        super().__init__()
        m: List[nn.Module] = []
        for i in range(2):
            atrous_rate = 1 if i == 0 else atrous
            conv_filter = utils.create_conv_layer(n_feats, n_feats, kernel_size, rate=atrous_rate, bias=True)
            m.append(conv_filter)

            if i == 0:
                m.append(utils.get_activation(act))

        self.body = nn.Sequential(*m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, scale: int, n_feats: int, act: str = "none", bias: bool = True) -> None:
        m: List[nn.Module] = []
        for _ in range(int(math.log(scale, 2))):
            m.append(utils.create_conv_layer(n_feats, 4 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(2))
            m.append(utils.get_activation(act))
        super(Upsampler, self).__init__(*m)


class EDSRDec(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, resblocks: int = 8, kernel_size: int = 3, tail: str = "none", channel_attention: bool = False) -> None:
        if tail != "conv":
          print("oops, abnormality")

        super().__init__()
        self.head = utils.create_conv_layer(in_ch, out_ch, 1)
        m_body: List[nn.Module] = [ResBlock(out_ch, kernel_size) for _ in range(resblocks)]
        m_body.append(utils.create_conv_layer(out_ch, out_ch, kernel_size))
        self.body = nn.Sequential(*m_body)
        self.tail = utils.create_conv_layer(out_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, features_to_fuse: torch.Tensor = torch.Tensor([0.0])) -> torch.Tensor:
        x = self.head(x)
        x = x + features_to_fuse
        x = self.body(x) + x
        x = self.tail(x)
        return x

class StrongPixDecoder(nn.Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.loss_fn = utils.DiscretizedMixLogisticLoss(rgb_scale=True)
        self.scale = scale
        self.loss_fn
        self.rgb_decs = nn.ModuleList([
            EDSRDec(
                3 * i, config.n_feats,
                resblocks=config.resblocks, tail="conv")
            for i in range(1, 4)
        ])
        self.mix_logits_prob_clf = nn.ModuleList([
            AtrousProbabilityClassifier(
                config.n_feats, C=3, K=config.K,
                num_params=utils._NUM_PARAMS_RGB)
            for _ in range(1, 4)
        ])
        self.feat_convs = nn.ModuleList([
            utils.create_conv_layer(config.n_feats, config.n_feats, 3) for _ in range(1, 4)
        ])

    def forward_probs(
        self, x: torch.Tensor, ctx: torch.Tensor
    ) -> Generator[utils.LogisticMixtureProbability, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        mode = "train" if self.training else "eval"
        pix_sum = x * 4  # N 3 H W, [0, 1020]
        xy_normalized = x / 127.5 - 1
        y_i = torch.tensor([], device=x.device)
        z: torch.Tensor = 0.  # type: ignore

        for i, (rgb_dec, clf, feat_conv) in enumerate(
            zip(self.rgb_decs, self.mix_logits_prob_clf, self.feat_convs)
        ):
            xy_normalized = torch.cat((xy_normalized, y_i / 127.5 - 1), dim=1)
            z = rgb_dec(xy_normalized, ctx)
            ctx = feat_conv(z)
            probs = clf(z)
            lower = torch.max(pix_sum - (3 - i) * 255, torch.tensor(0., device=x.device))
            upper = torch.min(pix_sum, torch.tensor(255., device=x.device))
            y_i = yield utils.LogisticMixtureProbability(
                f"{mode}/{self.scale}_{i}", i, probs, lower, upper
            )
            _, _, xH, xW = y_i.size()
            y_i = F.pad(y_i, [0, x.shape[-2] - xW, 0, x.shape[-1] - xH], mode="replicate")
            pix_sum -= y_i

        return pix_sum, ctx

    def forward(self, x: torch.Tensor, y: torch.Tensor, ctx: torch.Tensor) -> Tuple[List[utils.ZEDMetrics], torch.Tensor]:
        zed_metrics: List[utils.ZEDMetrics] = []
        y_slices = utils.group_2x2(y)
        gen = self.forward_probs(x, ctx)

        try:
            for i, y_slice in enumerate(y_slices):
                if i == 0:
                    lm_params = next(gen)
                else:
                    lm_params = gen.send(y_slices[i - 1])

                # print(f"LM: Name: {lm_params.name}, Level: {lm_params.pixel_index}, \
                #     Probs: {lm_params.probs.shape}, Upper: {lm_params.upper.shape}, \
                #     Lower: {lm_params.lower.shape}")

                logits = lm_params.probs

                # nll_map, entropy_map = get_metrics_from_logits(logits, y_slice)
                nll_map, entropy_map = self.loss_fn.get_zed_metrics(y_slice, logits)

                # print(f"NLL: {nll_map.shape}, Entropy: {entropy_map.shape}")

                # nll_map = self.loss_fn(y_slice, logits)
                # log_probs_all = get_log_prob_from_logits(logits, True)
                # entropy_map = calculate_entropy(log_probs_all)
                # true_pixel_indices = y_slice.long().unsqueeze(-1)
                # log_prob_true_pixel = torch.gather(log_probs_all, -1, true_pixel_indices)
                # nll_map = -log_prob_true_pixel.squeeze(-1)
                zed_metrics.append(utils.ZEDMetrics(nll_map, entropy_map))

        except StopIteration as e:
            _, new_ctx = e.value
            return zed_metrics, new_ctx

        raise RuntimeError("Generator finished without StopIteration.")


class Compressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        assert config.scale >= 0, config.scale

        self.ctx_upsamplers = nn.ModuleList([
            nn.Identity(),  # type: ignore
            *[Upsampler(scale=2, n_feats=config.n_feats)
              for _ in range(config.scale-1)]
        ] if config.scale > 0 else [])
        self.decs = nn.ModuleList([
            StrongPixDecoder(i) for i in range(config.scale)
        ])
        assert len(self.ctx_upsamplers) == len(self.decs), \
            f"{len(self.ctx_upsamplers)}, {len(self.decs)}"
        self.nets = nn.ModuleList([
            self.ctx_upsamplers, self.decs,
        ])

    def forward(self, x: torch.Tensor) -> Dict[int, utils.ZEDMetrics]:
        downsampled = utils.average_downsamples(x)

        # The list will store metrics from all levels, starting from the lowest res
        metrics_from_all_levels = {}
        ctx = 0.

        # This loop goes from lowest resolution decoder to highest
        for i, dec, ctx_upsampler, x_level, y_level, in zip(
                range(len(self.decs)), self.decs, self.ctx_upsamplers,
                downsampled[::-1], downsampled[-2::-1]):
            # print(f"Level {i} | x: {x_level.shape} | y: {y_level.shape}")
            # if type(ctx) == torch.Tensor:
            #     print(f"ctx (Tensor) shape (B, N_FEAT, H, W): {ctx.shape}")
            # else:
            #     print(f"ctx (Float) shape: {ctx}")

            ctx = ctx_upsampler(ctx)
            # if type(ctx) == torch.Tensor:
            #     print(f"ctx after upsampling (Tensor) shape (B, N_FEAT, H, W): {ctx.shape}")

            dec_metrics, ctx = dec(x_level, torch.round(y_level - 0.001), ctx)

            level_l = (len(self.decs) - 1) - i
            metrics_from_all_levels[level_l] = dec_metrics

        # print(metrics_from_all_levels.keys())
        return metrics_from_all_levels

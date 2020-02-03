import torch
from spatial_correlation_sampler import spatial_correlation_sample


def run_spatial_corr(rank):
    corr = spatial_correlation_sample(torch.ones(1, 512, 12, 27).to(f"cuda:{rank}"),
                                      torch.ones(1, 512, 12, 27).to(f"cuda:{rank}")).mean()
    print(corr)

run_spatial_corr(0)
run_spatial_corr(1)

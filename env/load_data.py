import torch
import numpy as np

from env.case_generate import CaseGenerator, ShopInfo


def load_rrpfsp(lines, shopinfo, device=torch.device('cpu')):
    """
    Load RRPFSP instances to structure data
    """

    # Read processing time information
    matrix_proc_time = []
    job_num = 0
    for line in lines:
        line_split = line.split()
        proc_time = []
        for i in line_split:
            x = float(i)
            if x > 0:
                proc_time.append(x)
        matrix_proc_time.append(proc_time)
        job_num += 1

    matrix_proc_time = torch.tensor(matrix_proc_time, device=device)
    matrix_proc_time = torch.cat((matrix_proc_time, torch.zeros((job_num, 1), device=device)), dim=1)

    return matrix_proc_time


if __name__ == "__main__":
    s = ShopInfo(job_min=10, job_max=100, paral_mas_num=[2, 2, 3, 2],
                 routing=[1, 2, 3, 4, 2, 3, 2, 1], proc_time_min=15, proc_time_max=200, trans_time=2)
    c = CaseGenerator(s, flag_doc=False)
    lines = c.get_case()[0]
    print(lines)
    print(load_rrpfsp(lines, s))



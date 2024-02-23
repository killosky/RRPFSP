
import numpy as np
import random
import os


class ShopInfo:
    """
    Production information of the shop, corresponding to a trained DRL agent.
    Including machine information, routing information, min and max processing time of each operation, etc.
    """
    def __init__(self, job_min, job_max, paral_mas_num, routing, proc_time_min, proc_time_max, trans_time,
                 trans_cap=2, buffer_cap=10, load_time=1, unload_time=1, proc_time_dev_para=20):

        self.job_min = job_min
        self.job_max = job_max
        self.paral_mas_num = paral_mas_num
        self.station_num = len(paral_mas_num)
        self.routing = routing
        self.ope_num = len(routing)
        self.proc_time_min = proc_time_min
        self.proc_time_max = proc_time_max
        self.trans_time = trans_time
        self.trans_cap = trans_cap
        self.buffer_cap = buffer_cap
        self.load_time = load_time
        self.unload_time = unload_time
        self.proc_time_dev_para = proc_time_dev_para


class CaseGenerator:
    """
    RRPFSP instance generator
    """
    def __init__(self, shop_info, path='.../data/', flag_doc=False):
        self.flag_doc = flag_doc    # Whether save the instance to a file
        self.path = path    # Instance save path (relative path)

        self.job_min = shop_info.job_min
        self.job_max = shop_info.job_max

        self.paral_mas_num = np.array(shop_info.paral_mas_num)
        self.mas_num = len(self.paral_mas_num)

        self.routing = np.array(shop_info.routing)
        self.ope_num = shop_info.ope_num

        self.proc_time_min = shop_info.proc_time_min
        self.proc_time_max = shop_info.proc_time_max
        self.proc_time_dev_para = shop_info.proc_time_dev_para
        self.trans_time = shop_info.trans_time
        self.trans_cap = shop_info.trans_cap
        self.buffer_cap = shop_info.buffer_cap
        self.load_time = shop_info.load_time
        self.unload_time = shop_info.unload_time

        self.num_jobs = None
        self.proc_time = None

    def proc_time_generate(self):
        """
        To make the instance more reasonable, the processing time of each operation is generated according to the rule
        """
        # Calculate the average number of machines for each operation
        num_mas_per_ope = np.zeros(self.ope_num)
        num_ope_per_mas = np.zeros(self.mas_num)
        for i in range(self.mas_num):
            num_ope_per_mas[i] = np.count_nonzero(self.routing == i+1) / self.paral_mas_num[i]

        for i in range(self.ope_num):
            num_mas_per_ope[i] = self.paral_mas_num[self.routing[i]-1] / num_ope_per_mas[self.routing[i]-1]

        # Calculate the average processing time of each operation
        proc_time_para = random.uniform(
            int(self.proc_time_min/min(num_mas_per_ope))+1, int(self.proc_time_max/max(num_mas_per_ope)))
        proc_time = np.array([proc_time_para * num_mas_per_ope[i] for i in range(self.ope_num)])

        # Allocate processing time of the operations on the same machine
        for i in range(self.mas_num):
            idx = np.where(self.routing == i+1)[0]
            if len(idx) > 1:
                proc_time_sum = sum(proc_time[idx])
                while True:
                    proc_time_regenerate = [random.randint(self.proc_time_min, self.proc_time_max) for _ in range(len(idx) - 1)]
                    proc_time_regenerate.append(proc_time_sum - sum(proc_time_regenerate))
                    if self.proc_time_min <= proc_time_regenerate[-1] <= self.proc_time_max:
                        proc_time[idx] = proc_time_regenerate
                        break

        # Add deviation to the processing time
        for i in range(self.ope_num):
            proc_time[i] += np.random.normal(0, proc_time[i]/self.proc_time_dev_para)
        proc_time = np.clip(proc_time, self.proc_time_min, self.proc_time_max)

        return proc_time

    def get_case(self, id_instance=0):
        """
        Generate RRPFSP instance generator
        :param id_instance: The instance index
        """
        # Generate job number
        self.num_jobs = random.randint(self.job_min, self.job_max)

        # Generate processing time
        self.proc_time = np.zeros(shape=(self.num_jobs, self.ope_num), dtype=float)
        for i in range(self.num_jobs):
            self.proc_time[i] = self.proc_time_generate()

        # Generate the instance
        lines = []
        for i in range(self.num_jobs):
            line = []
            for j in range(self.ope_num):
                line.append(self.proc_time[i][j])

            str_line = " ".join([str(val) for val in line])
            lines.append(str_line)

        # Save the instance to a file
        if self.flag_doc:
            if not os.path.exists(self.path):
                os.makedirs(self.path)

            doc = open(self.path + '{0}.rfs'.format(str.zfill(str(id_instance), 3),), 'a')
            for i in range(len(lines)):
                print(lines[i], file=doc)
            doc.close()

        return lines, self.num_jobs


def trans_time_generate(num_station, trans_time_ave, trans_time_dev):
    """
    Generate the transportation time between stations using average and deviation of the trans time
    :return: Transportation time matrix of the shop.
    """
    transtime_matrix = np.zeros((num_station+3, num_station+3), dtype=float)
    for i in range(num_station+3):
        for j in range(i+1, num_station+3):
            transtime = random.uniform(
                trans_time_ave * (1-trans_time_dev),
                trans_time_ave * (1+trans_time_dev)
            )
            transtime_matrix[i][j] = transtime
            transtime_matrix[j][i] = transtime

    return transtime_matrix


def job_num_detec(lines):
    num_job = 0
    for _ in lines:
        num_job += 1
    return num_job


if __name__ == "__main__":
    path = os.path.abspath('.') + "/data/"
    trans_time_matrix = trans_time_generate(7, 2, 0.2)
    s = ShopInfo(job_min=4, job_max=10, paral_mas_num=[2, 2, 3, 2],
                 routing=[1, 2, 3, 4, 2, 3, 2, 1], proc_time_min=15, proc_time_max=200, trans_time=trans_time_matrix)
    # c = CaseGenerator(s, path=path, flag_doc=True)

    # for i_instance in range(100):
    #     c.get_case(i_instance)

    c = CaseGenerator(s, flag_doc=False)
    print(c.get_case())


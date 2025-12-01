

class Model:
    def __init__(self):
        self.degree_state = 0
        self.degree_input = 0
        self.degree_disturbance = 0
        self.disturbance_type = ""
        self.d_para = None
        self.env = None

    @staticmethod
    def fx(x, u):
        raise NotImplementedError

    # def get_d_lb(self):
    #     return self.d_lb
    #
    # def get_d_ub(self):
    #     return self.d_ub





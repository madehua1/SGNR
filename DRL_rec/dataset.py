import torch.utils.data as data


class Userdata(data.Dataset):
    def __init__(self,users):
        """
        :param users: 输入用户    tensor()
        """
        super(Userdata, self).__init__()
        self.users = users

    def __len__(self):
        return self.users.shape[0]

    def __getitem__(self, index):
        return self.users[index]

import os
import shutil
import pickle as pkl

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

'''
路径管理工具，卢宁在华为项目中写的，可以删掉
删掉后把其他脚本里的路径改为正确路径
'''

def real_path_of(*path_in_repo):
    real_path = os.path.join(REPO_DIR, *path_in_repo)
    return real_path

def create_folder(*folder_path_in_repo):
    folder_dir = real_path_of(*folder_path_in_repo)
    if not os.path.isdir(folder_dir):
        os.makedirs(folder_dir)

def clean_folder(*folder_path_in_repo):
    folder_dir = real_path_of(*folder_path_in_repo)
    for filename in os.listdir(folder_dir):
        file_path = os.path.join(folder_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def save_pkl_in_repo(obj, *path_in_repo):
    full_path = real_path_of(*path_in_repo)
    save_pkl(obj, full_path)


def load_pkl_in_repo(*path_in_repo):
    full_path = real_path_of(*path_in_repo)
    return load_pkl(full_path)


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pkl.load(f)


def file_is_exist(*file_path_in_repo):
    file_path = real_path_of(*file_path_in_repo)
    return os.path.exists(file_path)


if __name__ == '__main__':
    tmp = real_path_of('test')
    print(tmp)

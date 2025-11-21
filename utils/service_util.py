# coding=utf-8
import os
import subprocess


from utils.env_util import EnvUtil


def get_app_env() -> str:
    """
    获取服务所在环境
    Returns:
        str: 服务所在环境
    """
    return os.getenv("APP_ENV", "")


def get_tenant() -> str:
    """
    获取服务所在环境
    Returns:
        str: 服务所在环境
    """
    return os.getenv("TENANT", "xz2_53")


def get_version_info_on_virtual() -> str:
    """
    获取部署在虚拟机上的git校验信息
    :return:
    """
    try:
        head_hash = subprocess.check_output('git rev-parse HEAD', shell=True).decode().strip()
        master_hash = subprocess.check_output('git rev-parse master', shell=True).decode().strip()
        origin_master_hash = subprocess.check_output('git rev-parse origin/master', shell=True).decode().strip()
        hash_verification = (head_hash == master_hash == origin_master_hash)

        head_message = subprocess.check_output('git log -1 --pretty=oneline', shell=True).decode().strip()
        head_message = ' '.join(head_message.split(' ')[1:])
        origin_master_message = subprocess.check_output('git log -1 --pretty=oneline origin/master',
                                                        shell=True).decode().strip()
        origin_master_message = ' '.join(origin_master_message.split(' ')[1:])
        git_params = {
            'verify_verbose': '一致' if hash_verification else 'ERROR!  当前服务代码非最新，请检查!',
            'head_hash': head_hash,
            'head_message': head_message,
            'origin_master_hash': origin_master_hash,
            'origin_master_message': origin_master_message,
        }

        git_info = '【版本验证】 {verify_verbose}\n\t' \
                   '【上线版本hash】 {head_hash}\n\t' \
                   '【远端主分支hash】 {origin_master_hash}\n\t' \
                   '【上线版本message】 {head_message}\n\t' \
                   '【远端主分支message】 {origin_master_message}\n\t' \
                   '【部署机器类型】 虚拟机\n\t'.format(**git_params)
        return git_info
    except Exception as e:
        return '\n\t无法获取git版本信息，【异常信息】{}\n\t'.format(str(e))


def get_version_info_on_cloud() -> str:
    """
    获取部署在云上的git校验版本信息
    :return:
    """
    try:
        head_hash = subprocess.check_output('git rev-parse HEAD', shell=True).decode().strip()
        head_message = subprocess.check_output('git log -1 --pretty=oneline', shell=True).decode().strip()
        head_message = ' '.join(head_message.split(' ')[1:])
        git_params = {
            'head_hash': head_hash,
            'head_message': head_message,
        }

        git_info = '【上线版本hash】 {head_hash}\n\t' \
                   '【上线版本message】 {head_message}\n\t' \
                   '【部署机器类型】 云\n\t'.format(**git_params)
        return git_info
    except Exception as e:
        return '\n\t无法获取git版本信息，【异常信息】{}\n\t'.format(str(e))


def get_version_info() -> str:
    is_on_cloud = EnvUtil.is_on_cloud()
    if is_on_cloud:
        return get_version_info_on_cloud()
    else:
        return get_version_info_on_virtual()


def get_service_started_info() -> str:
    """
    服务启动时，获取服务信息
    :return:
    """
    git_version_info = get_version_info()
    host_info = '【机器编号】 {}\n\t'.format(EnvUtil.get_hostname())
    start_info = 'start service!\n\t'
    service_info = '{}{}{}'.format(start_info, git_version_info, host_info)
    return service_info

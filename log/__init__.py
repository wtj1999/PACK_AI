# from configs.log_config import LOG_NAME_SERVICE, LOG_BAD_REQUEST, LOCAL_LOG
# from configs.monitor_config import RECIPIENTS, BAD_REQUEST_RECIPIENTS, LOCAL_RECIPIENTS
# from log.loggers import get_logger
# from utils.service_util import get_app_env
#
#
# env = get_app_env()
# mail_recipients = RECIPIENTS.get(env)
# request_mail_recipients = BAD_REQUEST_RECIPIENTS.get(env)
#
# logger = get_logger(
#     name=LOG_NAME_SERVICE,
#     fromaddr='caoxianfeng@gotion.com.cn',
#     toaddr=mail_recipients,
#     ignore_mail_env=False
# )
#
# request_logger = get_logger(
#     name=LOG_BAD_REQUEST,
#     fromaddr='caoxianfeng@gotion.com.cn',
#     toaddr=request_mail_recipients,
#     ignore_mail_env=False
# )
#
# # 本地电脑测试
# local_logger = get_logger(
#     name=LOCAL_LOG,
#     fromaddr='caoxianfeng@gotion.com.cn',
#     toaddr=LOCAL_RECIPIENTS['per'],
#     ignore_mail_env=True
# )

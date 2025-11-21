# # coding=utf-8
# import logging
# import os
# import socket
# import sys
# from datetime import datetime, timedelta
# from logging.handlers import SMTPHandler
#
# from configs.log_config import LOG_FILE, MAIL_CONFIG, CREDENTIALS
# from utils.env_util import EnvUtil
# from utils.service_util import get_app_env
#
#
# def get_local_name():
#     """
#     获取本机文件
#     :return: 主机名称
#     """
#     name = socket.gethostname()
#     return name
#
#
# def generate_log_file_path(file_name):
#     """
#     获取日志存储路径
#     :param file_name:
#     :return: 日志存储路径
#     """
#     cur_path = os.path.abspath(__name__)
#     dir_path = os.path.dirname(cur_path)
#     log_file_dir = os.path.join(dir_path, 'logs')
#     if not os.path.exists(log_file_dir):
#         os.mkdir(os.path.dirname(log_file_dir))
#     log_file_path = os.path.join(log_file_dir, file_name)
#     return log_file_path
#
#
# class BatchSMTPHandler(SMTPHandler):
#     """
#     邮件发送间隔不低于10s
#     """
#
#     def __init__(self, min_gaps_s=10, **kwargs):
#         SMTPHandler.__init__(self, **kwargs)
#         self.min_gaps_s = min_gaps_s
#         self.last_sent_at = datetime.now() - timedelta(0, min_gaps_s)
#
#     def emit(self, record):
#         now = datetime.now()
#         if now >= self.last_sent_at + timedelta(0, self.min_gaps_s):
#             try:
#                 self.last_sent_at = now
#                 SMTPHandler.emit(self, record)
#             except:
#                 pass
#
#
# class BufferingHandler(BatchSMTPHandler):
#     """
#     邮件超过异常数量时预警
#     1：日志累计5份，发送邮件
#     2：日志累计超过2份且距上一封邮件超过1小时，发送邮件
#     """
#
#     def __init__(self, capacity=5, max_gap_h=1, min_gap_s=10, **kwargs):
#         BatchSMTPHandler.__init__(self, min_gap_s, **kwargs)
#         self.capacity = capacity
#         self.max_gap_h = max_gap_h
#         self.latest_sent_at = datetime.now()
#         self.buffer = []
#
#     def shouldFlush(self):
#         now = datetime.now()
#         is_timeout = now > self.latest_sent_at + timedelta(0, hours=self.max_gap_h)
#         if (is_timeout and (len(self.buffer) >= 2)) or (len(self.buffer) >= self.capacity):
#             self.latest_sent_at = now
#             return True
#         return False
#
#     def emit(self, record):
#         self.buffer.append(record)
#         if self.shouldFlush():
#             error_info = ''
#             for item in self.buffer:
#                 error_info += "\n" + item.msg
#             record.msg = error_info
#             BatchSMTPHandler.emit(self, record)
#             self.flush()
#
#     def flush(self):
#         self.acquire()
#         try:
#             self.buffer.clear()
#         finally:
#             self.release()
#
#
# def get_logger(name, fromaddr=None, toaddr=None, ignore_mail_env=False):
#     """
#     获取日志句柄
#     :param name(str): 日志句柄名
#     :param fromaddr(str): 日志邮件发送者
#     :param toaddr(list): 日志邮件接收者
#     :return:
#         logging.Logger: 生成的日志实例
#     """
#     if not name:
#         name = 'HandlerLogger'
#
#     logger = logging.getLogger(name)
#     if getattr(logger, 'handlers', None):
#         mail_handler = [handler for handler in logger.handlers if isinstance(handler, BatchSMTPHandler)]
#         for hdl in mail_handler:
#             hdl.fromaddr = fromaddr
#             hdl.toaddr = toaddr
#         return logger
#
#     logger.setLevel(logging.DEBUG)
#
#     log_formatter = logging.Formatter(
#         fmt='%(asctime)s [%(levelname)s] %(name)s '
#             '%(filename)s[line:%(lineno)d] | request_ms=%(request_ms)s %(request_id)s %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
#
#     log_file_path = generate_log_file_path(LOG_FILE)
#
#     file_handler = logging.handlers.WatchedFileHandler(log_file_path, encoding='utf-8')
#     file_handler.setFormatter(log_formatter)
#     logger.addHandler(file_handler)
#
#     stdout_handler = logging.StreamHandler(sys.stdout)
#     stdout_handler.setFormatter(log_formatter)
#     logger.addHandler(stdout_handler)
#
#     is_in_mail_env = is_mail_env() if not ignore_mail_env else True
#     if is_in_mail_env and (toaddr is not None):
#         mail_handler = SMTPHandler(
#             mailhost=MAIL_CONFIG['mail_host'],
#             fromaddr=fromaddr,
#             toaddrs=toaddr,
#             subject='[CRITICAL] {} from {}'.format(name, get_local_name()),
#             credentials=CREDENTIALS['prod'],
#         )
#         mail_handler.setLevel(logging.CRITICAL)
#         mail_handler.setFormatter(log_formatter)
#         logger.addHandler(mail_handler)
#
#     return logger
#
#
# def is_mail_env():
#     env = get_app_env()
#     is_mail = os.environ.get("MAIL", "")
#     if is_mail == '0':
#         return False
#     return EnvUtil.is_prod_env(env) or EnvUtil.is_test_env(env)

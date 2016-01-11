#!/usr/bin/env python
import paramiko
import os

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.load_system_host_keys()
def get_host_server_info(ip_extranet):
    if ip_extranet == 1:
        host_ip = os.environ['ssh_extranet_host_ip']
        host_port = os.environ['ssh_extranet_host_port']
    else:
        host_ip = os.environ['ssh_intranet_host_ip']
        host_port = os.environ['ssh_intranet_host_port']
    host_username = os.environ['ssh_host_username']
    host_password = os.environ['ssh_host_password']
    return host_ip, host_port, host_username, host_password


def connect_ssh_and_execute_command(host_ip, host_port, host_username, host_password, command_to_execute):
    try:
   	if isinstance(host_port, basestring):
	    host_port = int(host_port)
        ssh.connect(host_ip, host_port, host_username, host_password)
        print 'connect successfully!'
        stdin, stdout, stderr = ssh.exec_command(command_to_execute)
        print 'execute successfully!'
        print stdout.readlines()
        ssh.close()    
    except Exception, error:
        print error


if __name__ == '__main__':
    ip_extranet = 1
    command_to_execute = 'pwd'
    host_ip, host_port, host_username, host_password = get_host_server_info(ip_extranet)
    connect_ssh_and_execute_command(host_ip, host_port, host_username, host_password, command_to_execute)

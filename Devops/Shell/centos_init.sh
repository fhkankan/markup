#!/bin/bash

set -x

rpm_install(){
  yum update -y
  yum install -y epel-release
  yum install -y man man-pages vim curl telnet zip unzip wget openssh-clients openssl lrzsz gcc gcc-c++ patch gdb lsof strace dos2unix sysstat iotop subversion git tig nfs-utils httpd-tools redis mysql redhat-lsb p7zip
  yum install -y glib2-devel zlib-devel bzip2-devel openssl-devel readline-devel mysql-devel libxml2-devel pcre-devel sqlite-devel libffi-devel
  yum install -y iftop htop eventlog libnfnetlink libnetfilter_conntrack conntrack-tools tmux tcpstat source-highlight numactl libnet libdbi ntpdate net-tools libselinux-python
}

centos_version(){
  yum install lsb -y
  version=$(lsb_release -rs)
  export version
}

base_system_tunning(){
  # set language US
  if ! grep 'LANGUAGE=en_US.UTF-8' /etc/profile >/dev/null; then
    echo '
export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8' >> /etc/profile
  fi

  # bash prompt
  if ! grep 'source /etc/bashrc' /etc/profile >/dev/null; then
    echo '
if [ $SHELL == /bin/bash ]; then
  source /etc/bashrc
fi' >> /etc/profile
  fi

  # Add serial tty
  if ! grep 'ttyS0' /etc/securetty >/dev/null; then
    echo 'ttyS0' >> /etc/securetty
  fi

  # disable ipv6
  if [ ! -f /etc/modprobe.d/net.conf ]; then
    touch /etc/modprobe.d/net.conf
  fi
  if ! grep 'options ipv6 disable=1' /etc/modprobe.d/net.conf >/dev/null; then
    echo "options ipv6 disable=1" >> /etc/modprobe.d/net.conf
  fi
  if ! grep 'alias net-pf-10 off' /etc/modprobe.d/net.conf >/dev/null; then
    echo "alias net-pf-10 off" >> /etc/modprobe.d/net.conf
  fi
  sed -i /NETWORKING_IPV6/cNETWORKING_IPV6=no /etc/sysconfig/network

  # motd text
  for i in motd issue issue.net; do
    if ! grep "Authorized users only. All activity may be monitored and reported." /etc/"$i" >/dev/null; then
      echo "Authorized users only. All activity may be monitored and reported." >> /etc/"$i"
    fi
  done

  # Add useful settings to /etc/sysctl.conf
  if [[ $version =~ ^6 ]]; then
    if ! grep '# Reboot a minute after an Oops' /etc/sysctl.conf >/dev/null; then
      echo '
# Reboot a minute after an Oops
kernel.panic = 60

# Syncookies make SYN flood attacks ineffective
net.ipv4.tcp_syncookies = 1

# Ignore bad ICMP
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1

# Disable ICMP Redirect Acceptance
net.ipv4.conf.all.accept_redirects = 0

# Enable IP spoofing protection, turn on source route verification
net.ipv4.conf.all.rp_filter = 0

# Log Spoofed Packets, Source Routed Packets, Redirect Packets
net.ipv4.conf.all.log_martians = 1

# Reply to ARPs only from correct interface (required for DSR load-balancers)
net.ipv4.conf.all.arp_announce = 2
net.ipv4.conf.all.arp_ignore = 1
fs.file-max = 1024000

net.ipv4.tcp_max_syn_backlog = 65536
net.core.netdev_max_backlog =  32768
net.core.somaxconn = 32768

net.core.wmem_default = 8388608
net.core.rmem_default = 8388608
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216

net.ipv4.tcp_timestamps = 0
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 2

net.ipv4.tcp_tw_recycle = 0
#net.ipv4.tcp_tw_len = 1
net.ipv4.tcp_tw_reuse = 1

net.ipv4.tcp_mem = 94500000 915000000 927000000
net.ipv4.tcp_max_orphans = 3276800

#net.ipv4.tcp_fin_timeout = 30
#net.ipv4.tcp_keepalive_time = 120
net.ipv4.ip_local_port_range = 1024  65535

vm.swappiness = 0
' >> /etc/sysctl.conf
    fi
  elif [[ $version =~ ^7 ]]; then
    if ! grep '# Reboot a minute after an Oops' /etc/sysctl.conf >/dev/null; then
      echo '
# Reboot a minute after an Oops
kernel.panic = 60

# Syncookies make SYN flood attacks ineffective
net.ipv4.tcp_syncookies = 1

# Ignore bad ICMP
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1

# Disable ICMP Redirect Acceptance
net.ipv4.conf.all.accept_redirects = 0

# Enable IP spoofing protection, turn on source route verification
net.ipv4.conf.all.rp_filter = 0

# Log Spoofed Packets, Source Routed Packets, Redirect Packets
net.ipv4.conf.all.log_martians = 1

# Reply to ARPs only from correct interface (required for DSR load-balancers)
net.ipv4.conf.all.arp_announce = 2
net.ipv4.conf.all.arp_ignore = 1
fs.file-max = 1024000

net.ipv4.tcp_max_syn_backlog = 65536
net.core.netdev_max_backlog =  32768
net.core.somaxconn = 32768

net.core.wmem_default = 8388608
net.core.rmem_default = 8388608
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216

net.ipv4.tcp_timestamps = 0
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 2

net.ipv4.tcp_tw_recycle = 1
#net.ipv4.tcp_tw_len = 1
net.ipv4.tcp_tw_reuse = 1

net.ipv4.tcp_mem = 94500000 915000000 927000000
net.ipv4.tcp_max_orphans = 3276800

#net.ipv4.tcp_fin_timeout = 30
#net.ipv4.tcp_keepalive_time = 120
net.ipv4.ip_local_port_range = 1024  65535

# disable ipv6
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1

# query size
kernel.msgmnb = 65536
kernel.msgmax = 65536

vm.swappiness = 0
' >> /etc/sysctl.conf
    fi
  fi

  # add more ip_conntrack_max
  if ! grep 'net.nf_conntrack_max = 655350' /etc/sysctl.conf >/dev/null; then
    echo 'net.nf_conntrack_max = 655350' >> /etc/sysctl.conf
  fi

  # increase for more connection
  #if ! grep 'net.ipv4.tcp_max_tw_buckets = 5000' /etc/sysctl.conf; then
  #  echo 'net.ipv4.tcp_max_tw_buckets = 5000' >>/etc/sysctl.conf
  #fi
  if ! grep 'net.ipv4.tcp_keepalive_time = 1200' /etc/sysctl.conf; then
    echo 'net.ipv4.tcp_keepalive_time = 1200' >> /etc/sysctl.conf
  fi
  if ! grep 'net.ipv4.tcp_keepalive_intvl = 15' /etc/sysctl.conf; then
    echo 'net.ipv4.tcp_keepalive_intvl = 15' >> /etc/sysctl.conf
  fi
  if ! grep 'net.ipv4.tcp_keepalive_probes = 5' /etc/sysctl.conf; then
    echo 'net.ipv4.tcp_keepalive_probes = 5' >> /etc/sysctl.conf
  fi
  if ! grep 'net.ipv4.tcp_fin_timeout = 30' /etc/sysctl.conf; then
    echo 'net.ipv4.tcp_fin_timeout = 30' >> /etc/sysctl.conf
  fi
  if ! grep 'net.netfilter.nf_conntrack_tcp_timeout_time_wait = 30' /etc/sysctl.conf; then
    echo 'net.netfilter.nf_conntrack_tcp_timeout_time_wait = 30' >> /etc/sysctl.conf
  fi
  sysctl -p

  # Add module ip_conntrack_ftp for iptables
  if ! grep 'ip_conntrack_ftp' /etc/sysconfig/iptables-config >/dev/null; then
    sed -i /IPTABLES_MODULES/s/\"$/\ ip_conntrack_ftp\"/ /etc/sysconfig/iptables-config
  fi

  ### add  ulimit for all user ###
  if ! grep '*                soft   nofile          409600' /etc/security/limits.conf >/dev/null; then
    echo '
*                soft   nofile          409600
*                hard   nofile          409600
' >> /etc/security/limits.conf
  fi

  # change open file ulimit started by root
  if ! grep 'root                soft   nofile          409600' /etc/security/limits.conf >/dev/null; then
    echo '
root                soft   nofile          409600
root                hard   nofile          409600
' >> /etc/security/limits.conf
  fi
  if ! grep 'ulimit -n 409600' /etc/init.d/functions >/dev/null; then
    sed -i '/export PATH/a  #\n# Increase File Descriptor \nulimit -n 409600' "/etc/init.d/functions"
  fi

  ### add history date ###
  if ! grep 'export HISTTIMEFORMAT="%F %T' /etc/bashrc >/dev/null; then
    echo 'export HISTTIMEFORMAT="%F %T "' >> /etc/bashrc
  fi

  ### change the command history #######
  sed -i '/^HISTSIZE=/c\HISTSIZE=10240' /etc/profile

  # setenforce 0
  setenforce 0
  sed -i 's/SELINUX=enforcing/SELINUX=disabled/' /etc/sysconfig/selinux
  sed -i 's/SELINUX=enforcing/SELINUX=disabled/' /etc/selinux/config
}

rpm_install
centos_version

echo "Starting base_system_tunning"
base_system_tunning
echo "Finished base_system_tunning"

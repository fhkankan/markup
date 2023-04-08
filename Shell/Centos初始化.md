# 服务器初始化

centos版本

```

```

## 环境处理

上传`centos_init.sh`

```shell
scp centos_init.sh root@xx.xx.xx:/root
```

登录执行

```shell
ssh root@xx.xx.xx
chmod a+x centos_init.sh
./centos_init.sh
```

python3.9安装

```shell
mkdir /opt/tgz
cd /opt/tgz
wget https://www.python.org/ftp/python/3.9.15/Python-3.9.15.tgz
tar xzvf Python-3.9.15.tgz
cd Python-3.9.15
./configure && make && make install
```

## 用户管理

```shell
useradd opuser
passwd opuser
cd /opt
mkdir soft
mkdir www
chown -R opuser:opuser /opt/soft /opt/www

sudo u+w /etc/sudoers
vim /etc/sudoers
# 在root ALL=(ALL) ALL添加
opuser ALL=(ALL) ALL
```

## 服务安装

mtk

```shell
su opuser
cd /opt/soft
git clone https://e.coding.net/eachplus/public-service/py-micro-toolkit.git

# root
cd  /opt/soft/py-micro-toolkit
pip3 install --upgrade pip wheel Cython
pip3 install --upgrade -r requirements.txt
./setup.sh
# opuser
./setup.sh
```

kepler（可选）

```shell
su opuser
cd /opt/soft
git clone https://e.coding.net/eachplus/public-service/kepler.git

# root
cd /opt/soft/kepler
./setup.sh

# 运维
service kepler restart
service kepler status
systemctl restart kepler.service
systemctl status kepler.service
```

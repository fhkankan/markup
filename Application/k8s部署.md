# K8S部署

## 准备

- ubuntu18

## 系统检查

节点主机名唯一，建议写入/etc/hosts

```
sudo hostnamectl set-hostname k8s-master
sudo hostnamectl set-hostname k8s-node
```

禁止swap分区

```
# 永久关闭
sudo vim /etc/fstab  # 注释掉swap那行，持久化生效
reboot，使用top查看Swap为0

# 暂时关闭
sudo swapoff -a

# 查看状态
sudo free -m
```

关闭防火墙

```
sudo ufw status

sudo ufw enable # 开启防火墙
sudo ufw disable # 关闭防火墙
```

## docker

版本对应

```
Kubernetes 1.15.2  -->Docker版本1.13.1、17.03、17.06、17.09、18.06、18.09
Kubernetes 1.15.1  -->Docker版本1.13.1、17.03、17.06、17.09、18.06、18.09
Kubernetes 1.15.0  -->Docker版本1.13.1、17.03、17.06、17.09、18.06、18.09
Kubernetes 1.14.5  -->Docker版本1.13.1、17.03、17.06、17.09、18.06、18.09
Kubernetes 1.14.4  -->Docker版本1.13.1、17.03、17.06、17.09、18.06、18.09
Kubernetes 1.14.3  -->Docker版本1.13.1、17.03、17.06、17.09、18.06、18.09
Kubernetes 1.14.2  -->Docker版本1.13.1、17.03、17.06、17.09、18.06、18.09
Kubernetes 1.14.1  -->Docker版本1.13.1、17.03、17.06、17.09、18.06、18.09
Kubernetes 1.14.0  -->Docker版本1.13.1、17.03、17.06、17.09、18.06、18.09
Kubernetes 1.13.5  -->Docker版本1.11.1、1.12.1、1.13.1、17.03、17.06、17.09、18.06
Kubernetes 1.13.5  -->Docker版本1.11.1、1.12.1、1.13.1、17.03、17.06、17.09、18.06
Kubernetes 1.13.4  -->Docker版本1.11.1、1.12.1、1.13.1、17.03、17.06、17.09、18.06
Kubernetes 1.13.3  -->Docker版本1.11.1、1.12.1、1.13.1、17.03、17.06、17.09、18.06
Kubernetes 1.13.2  -->Docker版本1.11.1、1.12.1、1.13.1、17.03、17.06、17.09、18.06
Kubernetes 1.13.1  -->Docker版本1.11.1、1.12.1、1.13.1、17.03、17.06、17.09、18.06
Kubernetes 1.13.0  -->Docker版本1.11.1、1.12.1、1.13.1、17.03、17.06、17.09、18.06
Kubernetes 1.12.*  -->Docker版本1.11.1、1.12.1、1.13.1、17.03、17.06、17.09、18.06
Kubernetes 1.11.*  -->Docker版本1.11.2到1.13.1、17.03
Kubernetes 1.10.*  -->Docker版本1.11.2到1.13.1、17.03
```

安装

```
sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common

sudo curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"

sudo apt-get -y update

sudo apt-cache madison docker-ce

sudo apt-get -y install docker-ce=18.06.3~ce~3-0~ubuntu

# sudo apt-get install docker-ce docker-ce-cli containerd.i0

sudo mkdir -p /etc/docker

sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://neb7qdm8.mirror.aliyuncs.com"]
}
EOF

sudo systemctl daemon-reload && sudo systemctl restart docker
```

## k8s安装

安装kubeadm kubeadm kubectl

```
curl -s https://mirrors.aliyun.com/kubernetes/apt/doc/apt-key.gpg | sudo apt-key add -
 
sudo tee /etc/apt/sources.list.d/kubernetes.list <<EOF 
deb https://mirrors.aliyun.com/kubernetes/apt/ kubernetes-xenial main
EOF
 
sudo apt-get update

sudo apt-cache madison kubelet kubectl kubeadm |grep '1.15.4-00'  

sudo apt install -y kubelet=1.15.4-00 kubectl=1.15.4-00 kubeadm=1.15.4-00 

# sudo apt-get install -y kubelet kubeadm kubectl

sudo apt-mark hold kubelet kubeadm kubectl

# 设置kubelet 开机自启动  
sudo systemctl enable kubelet
```

镜像拉取替换

```
# 查看kubeadm config所需镜像
kubeadm config images list

# 替换
#!/bin/bash
K8S_VERSION=v1.15.4
PAUSE_VERSION=3.1
ETCD_VERSION=3.3.10
DNS_VERSION=1.3.1
DASHBOARD_VERSION=v2.0.0

# 基本组件
docker pull mirrorgooglecontainers/kube-apiserver-amd64:$K8S_VERSION
docker pull mirrorgooglecontainers/kube-controller-manager-amd64:$K8S_VERSION
docker pull mirrorgooglecontainers/kube-scheduler-amd64:$K8S_VERSION
docker pull mirrorgooglecontainers/kube-proxy-amd64:$K8S_VERSION
docker pull mirrorgooglecontainers/etcd-amd64:$ETCD_VERSION
docker pull mirrorgooglecontainers/pause:$PAUSE_VERSION
docker pull coredns/coredns:$DNS_VERSION

# 修改tag
docker tag mirrorgooglecontainers/kube-apiserver-amd64:$K8S_VERSION k8s.gcr.io/kube-apiserver:$K8S_VERSION
docker tag mirrorgooglecontainers/kube-controller-manager-amd64:$K8S_VERSION k8s.gcr.io/kube-controller-manager:$K8S_VERSION
docker tag mirrorgooglecontainers/kube-scheduler-amd64:$K8S_VERSION k8s.gcr.io/kube-scheduler:$K8S_VERSION
docker tag mirrorgooglecontainers/kube-proxy-amd64:$K8S_VERSION k8s.gcr.io/kube-proxy:$K8S_VERSION
docker tag mirrorgooglecontainers/etcd-amd64:$ETCD_VERSION k8s.gcr.io/etcd:$ETCD_VERSION
docker tag mirrorgooglecontainers/pause:$PAUSE_VERSION k8s.gcr.io/pause:$PAUSE_VERSION
docker tag coredns/coredns:$DNS_VERSION k8s.gcr.io/coredns:$DNS_VERSION

#删除冗余的images
docker rmi mirrorgooglecontainers/kube-apiserver-amd64:$K8S_VERSION
docker rmi mirrorgooglecontainers/kube-controller-manager-amd64:$K8S_VERSION
docker rmi mirrorgooglecontainers/kube-scheduler-amd64:$K8S_VERSION
docker rmi mirrorgooglecontainers/kube-proxy-amd64:$K8S_VERSION
docker rmi mirrorgooglecontainers/etcd-amd64:$ETCD_VERSION
docker rmi mirrorgooglecontainers/pause:$PAUSE_VERSION
docker rmi coredns/coredns:$DNS_VERSION
```

## 安装master节点

### master初始化

```
# 单机版
sudo kubeadm init --pod-network-cidr 172.16.0.0/16 \
    --ignore-preflight-errors=Swap
    
# 多节点版
sudo kubeadm init \
--apiserver-advertise-address=172.16.35.18 \
--pod-network-cidr=10.244.0.0/16  \
--ignore-preflight-errors=Swap
    
#  可配置
--apiserver-advertise-address: k8s中的主要服务apiserver的部署地址，填自己的管理节点 ip
--image-repository: 拉取的 docker 镜像源，因为初始化的时候kubeadm会去拉 k8s 的很多组件来进行部署，所以需要指定国内镜像源，下不然会拉取不到镜像。
--pod-network-cidr: 这个是 k8s 采用的节点网络，因为我们将要使用flannel作为 k8s 的网络，所以这里填10.244.0.0/16就好
--kubernetes-version: 这个是用来指定你要部署的 k8s 版本的，一般不用填，不过如果初始化过程中出现了因为版本不对导致的安装错误的话，可以用这个参数手动指定。
--ignore-preflight-errors: 忽略初始化时遇到的错误，比如说我想忽略 cpu 数量不够 2 核引起的错误，就可以用--ignore-preflight-errors=CpuNum。错误名称在初始化错误时会给出来。
```

注意

```
# 异常时需重制
sudo kubeadm reset

# 正常时需要拷贝kubeadm join开头命令，用于安装工作节点使用
kubeadm join 172.16.35.16:6443 --token wukq2k.pcfkk3evl7suhvk2 \
    --discovery-token-ca-cert-hash sha256:0ef08dc96f876a8ec211991dddebe7e3f3d2a47f4e1d654ccce1de723d071797 
# 若是没有记录，可以
 kubeadm token create --print-join-command
```

### kubectl配置调用

```
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

kubectl get all
kubectl get nodes # 查看已加入的节点
kubectl get cs  # 查看集群状态
```

### flannel网络

```
kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml

kubectl get pods -A

grep -i image kube-flannel.yml  # 显示配置的image

sudo docker pull quay.io/coreos/flannel:v0.14.0
```

## salve加入网络

使用join加入

```
# 若是记录
sudo kubeadm join 192.168.56.11:6443 --token wbryr0.am1n476fgjsno6wa --discovery-token-ca-cert-hash sha256:7640582747efefe7c2d537655e428faa6275dbaff631de37822eb8fd4c054807
# 若无记录
kubeadm token create --print-join-command
sudo kubeadm join ...
```

在master中查看节点状态

```
kubectl get nodes
```

## 集群验证

在 Kubernetes 集群中创建一个 pod，验证是否正常运行，分别执行以下命令：

```
# 部署
kubectl create deployment nginx --image=nginx

# 服务公开
kubectl expose deployment nginx --port=80 --type=NodePort
```

查看

```
[root@k8s-master ~]# kubectl get pod,svc
NAME                         READY   STATUS    RESTARTS   AGE
pod/nginx-6799fc88d8-zx89s   1/1     Running   0          27m

NAME                 TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
service/kubernetes   ClusterIP   10.96.0.1       <none>        443/TCP        143m
service/nginx        NodePort    10.111.56.177   <none>        80:30880/TCP   27m
```

在节点下访问

```
http://192.168.184.137:30880
http://192.168.184.138:30880 ddd
```

## dashboard

- 安装

```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.0-beta4/aio/deploy/recommended.yaml

# 查看
kubectl get pods -A

kubectl get namespaces

kubectl cluster-info
```

- 创建账号

```shell
# 创建用户
kubectl create serviceaccount dashboard-admin -n kube-system
# 用户授权
kubectl create clusterrolebinding dashboard-admin --clusterrole=cluster-admin --serviceaccount=kube-system:dashboard-admin
# 获取用户Token
kubectl describe secrets -n kube-system $(kubectl -n kube-system get secret | awk '/dashboard-admin/{print $1}')  # 方法一
kubectl -n kubernetes-dashboard describe secret $(kubectl -n kubernetes-dashboard get secret | grep dashboard-admin | awk '{print $1}')  # 方法二
```

- 浏览器访问

```
https://172.16.35.16:6443/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
```

进去，输入token即可进入,注意：token的值一行，不要分行

异常：权限不足

```
# 配置
echo '
apiVersion: v1
kind: ServiceAccount
metadata:
  labels:
    k8s-app: kubernetes-dashboard
  name: dashboard-admin
  namespace: kubernetes-dashboard' > dashboard-admin.yaml 

# 执行
kubectl create -f ./dashboard-admin.yaml 

# 配置
echo '
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: dashboard-admin-bind-cluster-role
  labels:
    k8s-app: kubernetes-dashboard
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: ServiceAccount
  name: dashboard-admin
  namespace: kubernetes-dashboard' > dashboard-role.yaml

# 执行
kubectl create -f ./dashboard-role.yaml   

#重新获取token  
kubectl -n kubernetes-dashboard describe secret $(kubectl -n kubernetes-dashboard get secret | grep dashboard-admin | awk '{print $1}')

# 重新登陆
退出，重新输入token
```

## 身份认证

三种客户端身份认证

```
- HTTPS 证书认证：基于CA证书签名的数字证书认证
- HTTP Token认证：通过一个Token来识别用户
- HTTP Base认证：用户名+密码的方式认证（不用）
```

获取信息

```shell
# 获取host
kubectl config view --minify | grep server | cut -f 2- -d ":" | tr -d " "
kubectl config view | grep server  # 方法二

# 获取token
# 方法一：
kubectl get secret -nkube-system | grep admin  # 获取admin配置项目
kubectl describe secret xxx -nkube-system | grep token  # 获取配置项的token
# 方法二
kubectl describe secret $(kubectl get secret -n kube-system | grep ^admin-user | awk '{print $1}') -n kube-system | grep -E '^token'| awk '{print $2}'  # 拼接命令直接获取token
```

### token配置

- 直接命令行

见dashboard的创建账号

- YAML创建

创建YAML

```
# 创建服务账号
# admin-user.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: admin-user
  namespace: kube-system
  
# 绑定角色
# admin-user-role-binding.yaml
apiVersion: rbac.authorization.k8s.io/v1beta1
kind: ClusterRoleBinding
metadata:
  name: admin-user
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: ServiceAccount
  name: admin-user
  namespace: kube-system
```

执行命令

```
kubectl create -f admin-user.yaml
kubectl create -f  admin-user-role-binding.yaml
```

获取Token

```
kubectl describe secret $(kubectl get secret -n kube-system | grep ^admin-user | awk '{print $1}') -n kube-system | grep -E '^token'| awk '{print $2}'
```

### 制作证书

k8s默认启用了RBAC，并为未认证用户赋予了一个默认的身份：anonymous

对于API Server来说，它是使用证书进行认证的，我们需要先创建一个证书：

```
# 我们使用client-certificate-data和client-key-data生成一个p12文件，可使用下列命令：
# 生成client-certificate-data
grep 'client-certificate-data' ~/.kube/config | head -n 1 | awk '{print $2}' | base64 -d >> kubecfg.crt

# 生成client-key-data
grep 'client-key-data' ~/.kube/config | head -n 1 | awk '{print $2}' | base64 -d >> kubecfg.key

# 生成p12
openssl pkcs12 -export -clcerts -inkey kubecfg.key -in kubecfg.crt -out kubecfg.p12 -name "kubernetes-client"
```

## 其他

单节点k8s,默认pod不被调度在master节点

```
kubectl taint nodes --all node-role.kubernetes.io/master-  # 去污点，master节点可以被调度
```

重启服务恢复k8s

```
docker start $(docker ps -a | awk '{print $1}' |tail -n +2)
```

# 应用项目

## 使用YML部署

### 准备

dockerfile

```
FROM ubuntu:20.04

# 更改ubuntu源
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

RUN  apt-get update
# -y表示在交互中默认y
RUN  apt-get upgrade -y

#将时间区改为上海时间---东八区
RUN apt-get install -y apt-utils
RUN apt-get install -y tzdata
RUN echo "Asia/Shanghai" > /etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata

# Install python3
RUN  apt-get install -y python3

# Install pip
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# 安装python包
RUN pip3 install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ \
    # 数据库连接引擎
    sanic==20.12.0

 # 清理缓存
RUN  apt-get autoremove -y \
     && apt-get autoclean -y

# 复制代码
COPY . /opt/soft/k8s_demo
WORKDIR /opt/soft/k8s_demo
EXPOSE 8000
CMD ["python3", "main.py"]
```

镜像处理

```
# 首先在阿里云镜像容器后台创建空镜像
docker login --username=henr**** registry.cn-hangzhou.aliyuncs.com  

docker build -t k8s_demo:v0.1 .

docker tag [ImageId]  registry.cn-hangzhou.aliyuncs.com/henry_123/k8sdemo:v0.1
docker push registry.cn-hangzhou.aliyuncs.com/henry_123/k8sdemo:v0.1
```

### 模版

deployment

```yaml
apiVersion: extensions/v1beta1  # 指定api版本，此值必须在kubectl api-versions中  
kind: Deployment  # 指定创建资源的角色/类型   
metadata:  # 资源的元数据/属性 
  name: demo  # 资源的名字，在同一个namespace中必须唯一
  namespace: default # 部署在哪个namespace中
  labels:  # 设定资源的标签
    app: nginx
    version: v1
spec: # 资源规范字段
  replicas: 1 # 声明副本数目
  revisionHistoryLimit: 3 # 保留历史版本
  selector: # 选择器
    matchLabels: # 匹配标签
      app: nginx
      version: v1
  strategy: # 策略
    rollingUpdate: # 滚动更新
      maxSurge: 30% # 最大额外可以存在的副本数，可以为百分比，也可以为整数
      maxUnavailable: 30% # 示在更新过程中能够进入不可用状态的 Pod 的最大值，可以为百分比，也可以为整数
    type: RollingUpdate # 滚动更新策略
  template: # 模版
    metadata: # 资源的元数据/属性 
      annotations: # 自定义注解列表
        sidecar.istio.io/inject: "false" # 自定义注解名字
      labels: # 设定资源的标签
        app: nginx
        version: v1
    spec: # 资源规范字段
      containers:
      - name: nginx# 容器的名字   
        image: nginx:1.17.0 # 容器使用的镜像地址   
        imagePullPolicy: IfNotPresent # 每次Pod启动拉取镜像策略，三个选择 Always、Never、IfNotPresent
                                      # Always，每次都检查；
                                      # Never，每次都不检查（不管本地是否有）；
                                      # IfNotPresent，如果本地有就不检查，如果没有就拉取（手动测试时，已经打好镜像存在docker容器中时，
                                      #    使用存在不检查级别， 默认为每次都检查，然后会进行拉取新镜像，因镜像仓库不存在，导致部署失败）
        volumeMounts:		#文件挂载目录，容器内配置
        - mountPath: /data/		#容器内要挂载的目录
          name: share	    #定义的名字，需要与下面vloume对应
        resources: # 资源管理
          limits: # 最大使用
            cpu: 300m # CPU，1核心 = 1000m
            memory: 500Mi # 内存，1G = 1000Mi
          requests:  # 容器运行时，最低资源需求，也就是说最少需要多少资源容器才能正常运行
            cpu: 100m
            memory: 100Mi
        livenessProbe: # pod 内部健康检查的设置
          httpGet: # 通过httpget检查健康，返回200-399之间，则认为容器正常
            path: /healthCheck # URI地址
            port: 8080 # 端口
            scheme: HTTP # 协议
            # host: 127.0.0.1 # 主机地址
          initialDelaySeconds: 30 # 表明第一次检测在容器启动后多长时间后开始
          timeoutSeconds: 5 # 检测的超时时间
          periodSeconds: 30 # 检查间隔时间
          successThreshold: 1 # 成功门槛
          failureThreshold: 5 # 失败门槛，连接失败5次，pod杀掉，重启一个新的pod
        readinessProbe: # Pod 准备服务健康检查设置
          httpGet:
            path: /healthCheck
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 30
          timeoutSeconds: 5
          periodSeconds: 10
          successThreshold: 1
          failureThreshold: 5
      	#也可以用这种方法   
      	#exec: 执行命令的方法进行监测，如果其退出码不为0，则认为容器正常   
      	#  command:   
      	#    - cat   
      	#    - /tmp/health   
      	#也可以用这种方法   
      	#tcpSocket: # 通过tcpSocket检查健康  
      	#  port: number 
        ports:
          - name: http # 名称
            containerPort: 8080 # 容器开发对外的端口 
            protocol: TCP # 协议
      imagePullSecrets: # 镜像仓库拉取密钥
        - name: harbor-certification
      volumes:		#挂载目录在本机的路径
      - name: share	#对应上面的名字
        hostPath:
          path: /data	#挂载本机的路径
      affinity: # 亲和性调试
        nodeAffinity: # 节点亲和力
          requiredDuringSchedulingIgnoredDuringExecution: # pod 必须部署到满足条件的节点上
            nodeSelectorTerms: # 节点满足任何一个条件就可以
            - matchExpressions: # 有多个选项，则只有同时满足这些逻辑选项的节点才能运行 pod
              - key: beta.kubernetes.io/arch
                operator: In
                values:
                - amd64
```

svc

```yaml
apiVersion: v1 # 指定api版本，此值必须在kubectl api-versions中 
kind: Service # 指定创建资源的角色/类型 
metadata: # 资源的元数据/属性
  name: demo # 资源的名字，在同一个namespace中必须唯一
  namespace: default # 部署在哪个namespace中
  labels: # 设定资源的标签
    app: demo
spec: # 资源规范字段
  type: ClusterIP # ClusterIP 类型
  ports:
    - port: 8080 # service 端口
      targetPort: http # 容器暴露的端口
      protocol: TCP # 协议
      name: http # 端口名称
  selector: # 选择器
    app: demo
```

### 应用

应用文件

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: k8sdemo
  namespace: default
  labels:
    app: k8sdemo
    version: v1
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: k8sdemo
        version: v1
    spec:
      containers:
      - name: k8sdemo
        image: registry.cn-hangzhou.aliyuncs.com/henry_123/k8sdemo:v0.2
        imagePullPolicy: "IfNotPresent"
        ports:
        - containerPort: 8000
```

服务文件

```yaml
apiVersion: v1
kind: Service
metadata:
  name: k8sdemo
  namespace: default
  labels:
    app: k8sdemo
spec:
  type: NodePort
  ports:
    - port: 8000
  selector:
    app: k8sdemo
```

部署项目并应用

```
kubectl create -f k8s_demo.deployment.yml & kubectl create -f k8s_demo.service.yml
```

检查服务状态

```
kubectl get pods
kubectl get services
```

删除应用

```
kubectl delete -f k8s_demo.deployment.yml & kubectl delete -f k8s_demo.service.yml
```

## 使用heml

[官网](https://helm.sh/zh/docs/)

helm和k8s对应版本

```
Helm 版本	支持的 Kubernetes 版本
3.5.x	1.20.x - 1.17.x
3.4.x	1.19.x - 1.16.x
3.3.x	1.18.x - 1.15.x
3.2.x	1.18.x - 1.15.x
3.1.x	1.17.x - 1.14.x
3.0.x	1.16.x - 1.13.x
2.16.x	1.16.x - 1.15.x
2.15.x	1.15.x - 1.14.x
2.14.x	1.14.x - 1.13.x
2.13.x	1.13.x - 1.12.x
2.12.x	1.12.x - 1.11.x
2.11.x	1.11.x - 1.10.x
2.10.x	1.10.x - 1.9.x
2.9.x	1.10.x - 1.9.x
2.8.x	1.9.x - 1.8.x
2.7.x	1.8.x - 1.7.x
2.6.x	1.7.x - 1.6.x
2.5.x	1.6.x - 1.5.x
2.4.x	1.6.x - 1.5.x
2.3.x	1.5.x - 1.4.x
2.2.x	1.5.x - 1.4.x
2.1.x	1.5.x - 1.4.x
2.0.x	1.4.x - 1.3.x
```

### 安装

>  二进制

每个Helm [版本](https://github.com/helm/helm/releases)都提供了各种操作系统的二进制版本，这些版本可以手动下载和安装。

1. 下载 [需要的版本](https://github.com/helm/helm/releases)
2. 解压(`tar -zxvf helm-v3.0.0-linux-amd64.tar.gz`)
3. 在解压目中找到`helm`程序，移动到需要的目录中(`mv linux-amd64/helm /usr/local/bin/helm`)

```
v3.1.3的amd64版本
```

然后就可以执行客户端程序并 [添加稳定仓库](https://helm.sh/zh/docs/intro/quickstart/#初始化): `helm help`.

> 脚本

```
$ curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
$ chmod 700 get_helm.sh
$ ./get_helm.sh
```

> 管理包

```
// mac
brew install kubernetes-helm

// ubuntu
curl https://baltocdn.com/helm/signing.asc | sudo apt-key add -
sudo apt-get install apt-transport-https --yes
echo "deb https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm
```

- 其他

在master上创建账户

```shell
# 在部署 Tiller 之前，我们需要创建一个在集群范围内的超级用户角色来分配给它，以便它可以在任何命名空间中创建和修改 Kubernetes 资源。为了实现这一点，我们首先创建一个服务帐户，通过此方法，pod 在与服务帐户关联时，可以向 Kubernetes API 进行验证，以便能够查看、创建和修改资源。我们在 kube 系统名称空间中创建它

kubectl --namespace kube-system create serviceaccount tiller
```

在此服务帐户和群集角色之间创建绑定，顾名思义，该绑定会授予群集范围内的管理权限

```shell
kubectl create clusterrolebinding tiller \
    --clusterrole cluster-admin \
    --serviceaccount=kube-system:tiller
```

将 Helm Tiller 部署到 Kubernetes 集群，并使用所需的访问权限

```shell
helm init --service-account tiller
```

### 使用

- 常用命令

```
# 添加库
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add aliyun https://kubernetes.oss-cn-hangzhou.aliyuncs.com/charts  # 国内

# 查看库
helm repo list
helm search repo mysql
helm show chart aliyun/mysql
helm show all aliyun/mysql

# 删除库
helm repo remove aliyun

# 安装
helm install aliyun/mysql --generate-name  # 自动命名
helm install aliyun/mysql --name mysql  # 手动命名
helm install foo foo-0.1.1.tgz  # 本地chart压缩包
helm install foo path/to/foo  # 解压后的 chart 目录
helm install foo https://example.com/charts/foo-1.2.3.tgz  # 完整的URL


# 列出所有可被部署的版本
helm list
# 查看版本发布
helm ls

# 卸载
helm uninstall happy-panda --keep-history  # 记录版本
helm status 名字

# 升级
helm upgrade -f panda.yaml happy-panda bitnami/wordpress

# 查看升级是否生效
helm get values happy-panda

# 回滚
helm rollback happy-panda 1

# 帮助
helm help
helm get -h


# 创建自己的chart
helm create deis-workflow
# 在编辑 chart 时，验证格式
helm lint
# 编辑结束，打包
helm package deis-workflow
# 安装
helm install deis-workflow ./deis-workflow-0.1.0.tgz
# 上传
# 创建新目录，移动包，建立远程仓库index.yaml文件，同步工具或手动上传chart和index文件到chart仓库中
mkdir fantastic-charts
cp deis-workflow-0.1.0.tgz fantastic-charts/
helm repo index fantastic-charts --url https://fantastic-charts.storage.googleapis.com
# 添加一个新的chart到已有仓库中
helm repo index无痕重建index.yaml文件
```

- 创建自己的chart

创建新的heml定义

```
helm create k8sdemo
```
生成文件

```
k8sdemo/
 | -- charts/
 | -- templates/
 | Chart.yaml
 | values.yaml

# charts目录包含我们的新表所依赖的其他表（我们不会使用这个），templates 目录包含我们的 Helm 模板，Chart.yaml 包含图表的核心信息（例如名称和版本信息），values.yaml 包含用于呈现模板的默认值的信息（如果没有从命令行设置值）
```

改造yml

`Values.yaml`

```yaml
label: k8sdemo
port: 8000
replicas: 1
image:
  repository: registry.cn-hangzhou.aliyuncs.com/henry_123/k8sdemo:v0.2
  pullPolicy: IfNotPresent
```

`template/depolyment.yml`

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: {{.Values.label}}
  namespace: default
  labels:
    app: {{.Values.label}}
    version: v1
spec:
  replicas: {{.Values.replicas}}
  template:
    metadata:
      labels:
        app: {{.Values.label}}
        version: v1
    spec:
      containers:
      - name: {{.Values.label}}
        image: {{.Values.image.repository}}
        imagePullPolicy: {{.Values.image.pullPolicy}}
        ports:
        - containerPort: {{.Values.port}}
```

`service.yaml` 

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{.Values.label}}
  namespace: default
  labels:
    app: {{.Values.label}}
spec:
  type: NodePort
  ports:
    - port: {{.Values.port}}
  selector:
    app: {{.Values.label}}
```

安装

```
helm install k8sdemo --debug --dry-run

helm install k8sdemo --debug --name k8sdemo
```


# helm

[官方文档](https://helm.sh/zh/docs/)

## 概述

helm本质是一个`k8s`的包管理器，它具备如下能力：

简化部署 ：Helm允许使用单个命令轻松部署和管理应用程序，从而简化了整个部署过程；
高度可配置：Helm Charts提供了高度可配置的选项，可以轻松自定义和修改应用程序的部署配置；
版本控制 ：Helm允许管理应用程序的多个版本，从而轻松实现版本控制和回滚；
模板化：Helm Charts使用`YAML`模板来定义`Kubernetes`对象的配置，从而简化了配置过程，并提高了可重复性和可扩展性；
应用程序库：Helm具有应用程序库的概念，可以轻松地共享和重用Helm Charts，从而简化了多个应用程序的部署和管理；
插件系统：Helm拥有一个强大的插件系统，允许您扩展和定制Helm的功能，以满足特定的需求和要求。

## 工作流

![](.\images\heml流程.png)

**如上图所示，Helm的工作流程总结如下：**

1. 开发者首先创建并编辑chart的配置；
2. 接着打包并发布至Helm的仓库（Repository）；
3. 当管理员使用helm命令安装时，相关的依赖会从仓库下载；
4. 接着helm会根据下载的配置部署资源至k8s；

## 概念

| 概念       | 描述                                                         |
| ---------- | ------------------------------------------------------------ |
| helm       | 是一个命令行工具，用于本地开发及管理chart，chart仓库管理等   |
| Tiller     | 是 Helm 的服务端。Tiller 负责接收 Helm 的请求，与 k8s 的 apiserver 交互，根据chart 来生成一个 release 并管理 release |
| Chart      | 一个Helm包，其中包含了运行一个应用所需要的镜像、依赖和资源定义等，还可能包含Kubernetes集群中的服务定义，类似Homebrew中的formula、APT的dpkg或者Yum的rpm文件 |
| Repository | 存储Helm Charts的地方                                        |
| Release    | Chart在k8s上运行的Chart的一个实例，例如，如果一个MySQL Chart想在服务器上运行两个数据库，可以将这个Chart安装两次，并在每次安装中生成自己的Release以及Release名称。 |
| Value      | Helm Chart的参数，用于配置Kubernetes对象                     |
| Template   | 使用Go模板语言生成Kubernetes对象的定义文件                   |
| Namespace  | Kubernetes中用于隔离资源的逻辑分区                           |

## 使用

### 命令汇总

控制台使用 `helm --help`即可查看

```shell
Usage:
  helm [command]

Available Commands:
  completion  generate autocompletion scripts for the specified shell
  create      create a new chart with the given name
  dependency  manage a chart's dependencies
  env         helm client environment information
  get         download extended information of a named release
  help        Help about any command
  history     fetch release history
  install     install a chart
  lint        examine a chart for possible issues
  list        list releases
  package     package a chart directory into a chart archive
  plugin      install, list, or uninstall Helm plugins
  pull        download a chart from a repository and (optionally) unpack it in local directory
  push        push a chart to remote
  registry    login to or logout from a registry
  repo        add, list, remove, update, and index chart repositories
  rollback    roll back a release to a previous revision
  search      search for a keyword in charts
  show        show information of a chart
  status      display the status of the named release
  template    locally render templates
  test        run tests for a release
  uninstall   uninstall a release
  upgrade     upgrade a release
  verify      verify that a chart at the given path has been signed and is valid
  version     print the client version information
```

常用

```shell
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

### 安装helm

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

### 创建chart	

命令行创建chart

```	
heml create wordpress	
```

生成目录包含描述应用程序的文件和目录，包括`Chart.yaml`、`values.yaml`、`templates`目录等；

```
wordpress               - chart 包目录名
├── charts              - 依赖的子包目录，里面可以包含多个依赖的chart包
├── Chart.yaml          - chart定义，可以定义chart的名字，版本号信息。
├── templates           - k8s配置模版目录， 我们编写的k8s配置都在这个目录， 除了NOTES.txt和下划线开头命名的文件，其他文件可以随意命名。
│   ├── deployment.yaml
│   ├── _helpers.tpl    - 下划线开头的文件，helm视为公共库定义文件，主要用于定义通用的子模版、函数等，helm不会将这些公共库文件的渲染结果提交给k8s处理。
│   ├── ingress.yaml
│   ├── NOTES.txt       - chart包的帮助信息文件，执行helm install命令安装成功后会输出这个文件的内容。
│   └── service.yaml
└── values.yaml         - chart包的参数配置文件，模版可以引用这里参数。
```

### 编辑yml

- `Chart.yaml`

模板及注释

```yaml
apiVersion: chart API 版本 （必需）  #必须有
name: chart名称 （必需）     # 必须有 
version: 语义化2 版本（必需） # 必须有

kubeVersion: 兼容Kubernetes版本的语义化版本（可选）
description: 一句话对这个项目的描述（可选）
type: chart类型 （可选）
keywords:
  - 关于项目的一组关键字（可选）
home: 项目home页面的URL （可选）
sources:
  - 项目源码的URL列表（可选）
dependencies: # chart 必要条件列表 （可选）
  - name: chart名称 (nginx)
    version: chart版本 ("1.2.3")
    repository: （可选）仓库URL ("https://example.com/charts") 或别名 ("@repo-name")
    condition: （可选） 解析为布尔值的yaml路径，用于启用/禁用chart (e.g. subchart1.enabled )
    tags: # （可选）
      - 用于一次启用/禁用 一组chart的tag
    import-values: # （可选）
      - ImportValue 保存源值到导入父键的映射。每项可以是字符串或者一对子/父列表项
    alias: （可选） chart中使用的别名。当你要多次添加相同的chart时会很有用

maintainers: # （可选） # 可能用到
  - name: 维护者名字 （每个维护者都需要）
    email: 维护者邮箱 （每个维护者可选）
    url: 维护者URL （每个维护者可选）

icon: 用做icon的SVG或PNG图片URL （可选）
appVersion: 包含的应用版本（可选）。不需要是语义化，建议使用引号
deprecated: 不被推荐的chart （可选，布尔值）
annotations:
  example: 按名称输入的批注列表 （可选）.
```

- `values.yaml`

包含应用程序的默认配置值，如：

```yaml
image:
  repository: nginx
  tag: '1.19.8'
```

- `templates/demployment.yaml`

在模板中引入`values.yaml`里的配置，在模板文件`deployment.yaml`中可以通过` .Values`对象访问到，如

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-helm-{{ .Values.image.repository }}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx-helm
  template:
    metadata:
      labels:
        app: nginx-helm
    spec:
      containers:
      - name: nginx-helm
        image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
        ports:
        - containerPort: 80
          protocol: TCP
```

### 打包chart

使用helm package命令将Chart打包为一个tarball文件，将生成一个名为`wordpress-0.1.0.tgz`的`tarball`文件

```shell
helm package wordpress
```

### 发布chart

将打包好的Chart发布到一个Helm Repository中。可以使用helm repo add命令添加一个Repository，然后使用helm push命令将Chart推送到Repository中

```shell
helm repo add myrepo https://example.com/charts
helm push wordpress-0.1.0.tgz myrepo
```

### 安装release

使用helm install命令安装Chart的Release，可以通过命令行选项或指定values.yaml文件来配置Release

```
helm install mywordpress myrepo/wordpress
```

这将在`Kubernetes`集群中创建一个名为`mywordpress`的`Release`，包含`WordPress`应用程序和`MySQL`数据库。

### 管理release

使用`helm ls`命令查看当前运行的Release列表

```
helm upgrade mywordpress myrepo/wordpress --set image.tag=5.7.3-php8.0-fpm-alpine
```

这将升级 `mywordpress` 的`WordPress`应用程序镜像版本为`5.7.3-php8.0-fpm-alpine`

可以使用`helm rollback`命令回滚到先前版本，如回滚`mywordpress`的版本到1

```
helm rollback mywordpress 1
```

### 更新chart

在应用程序更新时，可以更新Chart配置文件和模板，并使用helm package命令重新打包Chart。然后可以使用helm upgrade命令升级已安装的Release，可以按照以下步骤更新Chart：

```
在本地编辑Chart配置或添加新的依赖项；
使用helm package命令打包新的Chart版本；
使用helm push命令将新的Chart版本推送到Repository中；
使用helm repo update命令更新本地或远程的Helm Repository；
使用helm upgrade命令升级现有Release到新的Chart版本。
```

将升级mywordpress的Chart版本到0.2.0，其中包括新的配置和依赖项。

```
helm upgrade mywordpress myrepo/wordpress --version 0.2.0
```

将删除名为mywordpress的Release，同时删除WordPress应用程序和MySQL数据库

```
helm uninstall mywordpress
```

将删除名为mywordpress的Release，并删除与之相关的所有PersistentVolumeClaim

```
helm uninstall mywordpress --delete-data
```

## 执行安装顺序

Helm按照以下顺序安装资源

```
Namespace
NetworkPolicy
ResourceQuota
LimitRange
PodSecurityPolicy
PodDisruptionBudget
ServiceAccount
Secret
SecretList
ConfigMap
StorageClass
PersistentVolume
PersistentVolumeClaim
CustomResourceDefinition
ClusterRole
ClusterRoleList
ClusterRoleBinding
ClusterRoleBindingList
Role
RoleList
RoleBinding
RoleBindingList
Service
DaemonSet
Pod
ReplicationController
ReplicaSet
Deployment
HorizontalPodAutoscaler
StatefulSet
Job
CronJob
Ingress
APIService
```

Helm 客户端不会等到所有资源都运行才退出，可以使用 helm status 来追踪 release 的状态，或是重新读取配置信息

```shell
helm status mynginx
```


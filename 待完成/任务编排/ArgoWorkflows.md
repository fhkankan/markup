# ArgoWorkflows

[Argo Workflows](https://link.zhihu.com/?target=https%3A//github.com/argoproj/argo-workflows) 是一个开源的**云原生工作流引擎**，用于在 Kubernetes 上编排并行作业。Argo 工作流作为Kubernetes CRD 实现。

- 定义工作流，其中工作流中的每个步骤都是一个容器。
- 将多步骤工作流建模为一系列任务，或使用 DAG 来捕获任务之间的依赖关系图。
- 使用 Argo 可以在很短的时间内在 Kubernetes 上轻松运行机器学习或数据处理的计算密集型作业

一句话描述：**ArgoWorkflow 是一个用于在 Kubernetes 上编排并行作业的开源云原生工作流引擎**。

## 安装

[安装文档](https://github.com/argoproj/argo-workflows/releases/)

cli

```
# Detect OS
ARGO_OS="darwin"
if [[ uname -s != "Darwin" ]]; then
  ARGO_OS="linux"
fi

# Download the binary
curl -sLO "https://github.com/argoproj/argo-workflows/releases/download/v3.6.2/argo-$ARGO_OS-amd64.gz"

# Unzip
gunzip "argo-$ARGO_OS-amd64.gz"

# Make binary executable
chmod +x "argo-$ARGO_OS-amd64"

# Move binary to path
mv "./argo-$ARGO_OS-amd64" /usr/local/bin/argo

# Test installation
argo version
```

controller and server

```
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.6.2/install.yaml
```

## CLI

```
argo submit hello-world.yaml    # submit a workflow spec to Kubernetes
argo list                       # list current workflows
argo get hello-world-xxx        # get info about a specific workflow
argo logs hello-world-xxx       # print the logs from a workflow
argo delete hello-world-xxx     # delete workflow
```


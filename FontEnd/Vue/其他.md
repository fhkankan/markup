# 其他

## 文件下载

- a标签

```

```

- blob

```vue
<template>
	<el-button @click="downloadUrl">导出Excel</el-button>        
</template>
<script>
import api from "../../api/api.js";
import axios from "axios";
export default {
  data() {
    return {
		pageSize:30,
	}
  },
  created() {},
  computed: {},
  mounted() {},
  methods: {
    downloadUrl() {
      // console.log(api.serverUrl);
      let params = {
        pageNum: "1",
        pageSize: this.pageSize,
        companyId: JSON.parse(sessionStorage.getItem("companyId"))
      };
      this.download_accountsDetails_info(params);
    },
    download_accountsDetails_info(params) {
      return new Promise((resolve, reject) => {
        axios
          .get(api.serverUrl + "/order/exportOrder", {
            params: params,
            // 1.首先设置responseType对象格式为 blob:
            responseType: "blob" 
          })
          .then(
            res => {
              //resolve(res)
              // 2.获取请求返回的response对象中的blob 设置文件类型，这里以excel为例
              let blob = new Blob([res.data], {
                type: "application/vnd.ms-excel"
              }); 
              // 3.创建一个临时的url指向blob对象
              let url = window.URL.createObjectURL(blob); 

              // 4.创建url之后可以模拟对此文件对象的一系列操作，例如：预览、下载
              let a = document.createElement("a");
              a.href = url;
              a.download = "导出表格.xlsx";
              a.click();
              // 5.释放这个临时的对象url
              window.URL.revokeObjectURL(url);
            },
            err => {
              resolve(err.response);
            }
          )
          .catch(error => {
            reject(error);
          });
      });
    },
};
</script>
<style>
</style>
```


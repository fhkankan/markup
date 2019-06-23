# 使用IrisSkin换肤

IrisSkin 是为Microsoft Visual Studio dotNET开发的最易用的界面增强dotNET(WinForm)组件包。能完全自动的为应用程序添加支持换肤功能。

## 文件

1. IrisSkin4.dll - 544 KB
2. 各种 .ssk 格式的皮肤文件（一般在网上搜的是13个皮肤的压缩包）

## 使用

- 打开IDE，引入动态库

工具/选择工具箱项/浏览/引入动态库IrisSkin4

- 工具箱创建选项卡，调用动态库

工具箱/空白处右键/添加选项卡/输入选项卡名字/右键新建的选项卡/选择项/筛选器中输入之前引入动态库IrisSkin4的名称/确定

工具箱中会出现一个新的皮肤插件，把皮肤插件拖到窗体中，会在窗体下方显示皮肤插件图标

- 选择皮肤

在需要的各种皮肤文件夹移动到建立的项目的bin目录下的debug文件夹下

```c#
public partial class Form1 : Form
{
    Sunisoft.IrisSkin.SkinEngine SkinEngine = new Sunisoft.IrisSkin.SkinEngine();
    public Form1()
    {
        InitializeComponent();
    }
    private void Form1_Load(object sender, EventArgs e)
    {
				// 加载确定的皮肤文件
				SkinEngine.SkinFile = Application.StartupPath + @"/Skins/mp10.ssk";
    }
}
```

## 测试代码

```c#
public partial class Form1 : Form
{
    Sunisoft.IrisSkin.SkinEngine SkinEngine = new Sunisoft.IrisSkin.SkinEngine();
    List<string> Skins;
    public Form1()
    {
        InitializeComponent();
    }
    private void Form1_Load(object sender, EventArgs e)
    {
        //加载所有皮肤列表
        Skins = Directory.GetFiles(Application.StartupPath + @"\IrisSkin4\Skins\", "*.ssk").ToList();
        Skins.ForEach(x =>
        {
        		// 将皮肤类表加载到dataGridView1控件的行数据
            dataGridView1.Rows.Add(Path.GetFileNameWithoutExtension(x));
        });
    }
    //选择皮肤并使用
    private void dataGridView1_CellDoubleClick(object sender, DataGridViewCellEventArgs e)
    {
        if (dataGridView1.CurrentRow != null)
        {
            //加载皮肤
            SkinEngine.SkinFile = Skins[dataGridView1.CurrentRow.Index];
            SkinEngine.Active = true;
        }
    }
    //打开 MessageBox 对话框
    private void BtMessageBox_Click(object sender, EventArgs e)
    {
        MessageBox.Show("MessageBoxMessageBoxMessageBoxMessageBox");
    }
    //打开测试窗口
    private void BtForm2_Click(object sender, EventArgs e)
    {
        new Form2().Show();
    }
    private void BtNormal_Click(object sender, EventArgs e)
    {
        //还原到默认皮肤
        SkinEngine.Active = false;
    }
}
```
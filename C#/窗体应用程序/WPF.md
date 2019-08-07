# 代码触发Button点击事件

先定义Button按钮并绑定事件。

```c#
public void test()
{
    Button btn = new Button();
    btn.Click += Btn_Click;
}

private void Btn_Click(object sender, RoutedEventArgs e)
{
     Console.WriteLine("点击了按钮！");
}
```

已经定义好了事件后，不点击按钮，如何靠代码动态触发按钮的点击事件？

方法一：

```c#
ButtonAutomationPeer peer = new ButtonAutomationPeer(someButton);
IInvokeProvider invokeProv = peer.GetPattern(PatternInterface.Invoke) as IInvokeProvider;
invokeProv.Invoke();
```

方法二：更优雅的方式

```c#
someButton.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
```

 
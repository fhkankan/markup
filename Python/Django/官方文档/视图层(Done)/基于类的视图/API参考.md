# 内置基于类视图API

基于类的视图API参考。有关入门资料，请参见[基于类的视图](https://yiyibooks.cn/__trs__/qy/django2/topics/class-based-views/index.html)主题指南

- Base views
  - [`View`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/base.html#view)
  - [`TemplateView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/base.html#templateview)
  - [`RedirectView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/base.html#redirectview)
- Generic display views
  - [`DetailView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-display.html#detailview)
  - [`ListView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-display.html#listview)
- Generic editing views
  - [`FormView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-editing.html#formview)
  - [`CreateView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-editing.html#createview)
  - [`UpdateView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-editing.html#updateview)
  - [`DeleteView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-editing.html#deleteview)
- Generic date views
  - [`ArchiveIndexView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-date-based.html#archiveindexview)
  - [`YearArchiveView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-date-based.html#yeararchiveview)
  - [`MonthArchiveView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-date-based.html#montharchiveview)
  - [`WeekArchiveView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-date-based.html#weekarchiveview)
  - [`DayArchiveView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-date-based.html#dayarchiveview)
  - [`TodayArchiveView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-date-based.html#todayarchiveview)
  - [`DateDetailView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-date-based.html#datedetailview)
- Class-based views mixins
  - Simple mixins
    - [`ContextMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-simple.html#contextmixin)
    - [`TemplateResponseMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-simple.html#templateresponsemixin)
  - Single object mixins
    - [`SingleObjectMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-single-object.html#singleobjectmixin)
    - [`SingleObjectTemplateResponseMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-single-object.html#singleobjecttemplateresponsemixin)
  - Multiple object mixins
    - [`MultipleObjectMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-multiple-object.html#multipleobjectmixin)
    - [`MultipleObjectTemplateResponseMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-multiple-object.html#multipleobjecttemplateresponsemixin)
  - Editing mixins
    - [`FormMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-editing.html#formmixin)
    - [`ModelFormMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-editing.html#modelformmixin)
    - [`ProcessFormView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-editing.html#processformview)
    - [`DeletionMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-editing.html#deletionmixin)
  - Date-based mixins
    - [`YearMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-date-based.html#yearmixin)
    - [`MonthMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-date-based.html#monthmixin)
    - [`DayMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-date-based.html#daymixin)
    - [`WeekMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-date-based.html#weekmixin)
    - [`DateMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-date-based.html#datemixin)
    - [`BaseDateListView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-date-based.html#basedatelistview)
- Class-based generic views - flattened index
  - Simple generic views
    - [`View`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#view)
    - [`TemplateView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#templateview)
    - [`RedirectView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#redirectview)
  - Detail Views
    - [`DetailView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#detailview)
  - List Views
    - [`ListView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#listview)
  - Editing views
    - [`FormView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#formview)
    - [`CreateView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#createview)
    - [`UpdateView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#updateview)
    - [`DeleteView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#deleteview)
  - Date-based views
    - [`ArchiveIndexView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#archiveindexview)
    - [`YearArchiveView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#yeararchiveview)
    - [`MonthArchiveView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#montharchiveview)
    - [`WeekArchiveView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#weekarchiveview)
    - [`DayArchiveView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#dayarchiveview)
    - [`TodayArchiveView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#todayarchiveview)
    - [`DateDetailView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/flattened-index.html#datedetailview)



## Specification

基于类的视图所服务的每个请求都具有独立的状态。因此，将状态变量存储在实例上是安全的（即`self.foo = 3`是线程安全的操作）。

使用`as_view()`类方法将基于类的视图部署到URL模式中：

```python
urlpatterns = [
    path('view/', MyView.as_view(size=42)),
]
```

> 带有视图参数的线程安全
传递给视图的参数在视图的每个实例之间共享。这意味着您不应将列表，字典或任何其他可变对象用作视图的参数。如果您这样做并且共享对象被修改，则一个用户访问您的视图的操作可能会对随后访问同一视图的用户产生影响。

传递给`as_view()`的参数将分配给用于服务请求的实例。使用前面的示例，这意味着MyView上的每个请求都可以使用`self.size`。参数必须与类中已经存在的属性相对应（在hasattr检查中返回True）。


## 基本视图与通用视图

可以将基于基类的视图视为父视图，这些视图可以自己使用或从其继承。它们可能无法提供项目所需的全部功能，在这种情况下，有些Mixins扩展了基本视图的功能。

Django的通用视图是基于这些基本视图构建的，并被开发为通用用法（如显示对象的详细信息）的快捷方式。它们采用了视图开发中发现的某些常见习语和模式，并对它们进行了抽象，以便您可以快速编写数据的通用视图而不必重复自己的工作。

大多数通用视图都需要queryset键，这是一个QuerySet实例。请参阅进行查询以获取有关QuerySet对象的更多信息。

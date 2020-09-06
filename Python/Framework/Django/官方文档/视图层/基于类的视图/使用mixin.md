# 使用mixin与基于类的视图

> 注意
这是一个进阶的话题。 需要建立在了解 Django’s class-based views的基础上。

Django的基于类的视图提供了许多功能，但是你可能只想使用其中的一部分。 例如，您可能需要编写一个视图来呈现模板以进行HTTP响应，但不能使用TemplateView；也许您只需要在POST上呈现模板，GET完全执行其他操作。 虽然你可以直接使用TemplateResponse，但是这将导致重复的代码。

由于这些原因，Django 提供许多Mixin，它们提供更细致的功能。 例如，渲染模板封装在TemplateResponseMixin 中。 Django 参考手册包含full documentation of all the mixins。

## 上下文和模板响应
在基于类的视图中使用模板具有一致的接口，有两个Mixin 起了核心的作用。

- `TemplateResponseMixin`

每个返回[`TemplateResponse`](https://yiyibooks.cn/__trs__/qy/django2/ref/template-response.html#django.template.response.TemplateResponse)的内置视图都将调用`TemplateResponseMixin`提供的[`render_to_response()`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-simple.html#django.views.generic.base.TemplateResponseMixin.render_to_response)方法。 大部分时间都会为你调用它（例如，它由[`TemplateView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/base.html#django.views.generic.base.TemplateView)和[`DetailView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-display.html#django.views.generic.detail.DetailView)实现的`get()`方法调用。 ）；同样地，你不太可能需要覆盖它，尽管如果你希望你的响应返回一些不通过Django模板呈现的东西，那么你就会想要这样做。 有关此示例，请参阅[JSONResponseMixin示例](https://yiyibooks.cn/__trs__/qy/django2/topics/class-based-views/mixins.html#jsonresponsemixin-example)。

`render_to_response()`本身调用[`get_template_names()`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-simple.html#django.views.generic.base.TemplateResponseMixin.get_template_names)，默认情况下只会在基于类的视图中查找[`template_name`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-simple.html#django.views.generic.base.TemplateResponseMixin.template_name)；另外两个mixins（[`SingleObjectTemplateResponseMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-single-object.html#django.views.generic.detail.SingleObjectTemplateResponseMixin)和[`MultipleObjectTemplateResponseMixin`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-multiple-object.html#django.views.generic.list.MultipleObjectTemplateResponseMixin)）会覆盖它，以便在处理实际对象时提供更灵活的默认值。

- `ContextMixin`

每个需要上下文数据的内置视图，例如渲染模板（包括上面的`TemplateResponseMixin`），都应该调用[`get_context_data()`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-simple.html#django.views.generic.base.ContextMixin.get_context_data)传递他们想要确保的任何数据在那里作为关键字参数。 `get_context_data()`返回一个字典；在`ContextMixin`中，它只返回其关键字参数，但通常会覆盖它以向字典中添加更多成员。 您还可以使用[`extra_context`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins-simple.html#django.views.generic.base.ContextMixin.extra_context)属性。

## 构建Django的通用类视图

让我们看一下Django的两个基于类的通用视图是如何用mixins构建的，提供离散功能。 我们将考虑[`DetailView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-display.html#django.views.generic.detail.DetailView)，它呈现一个对象的“细节”视图，以及[`ListView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-display.html#django.views.generic.list.ListView)，它将呈现一个对象列表，通常来自查询集，并且可选择分页他们。 这将向我们介绍四个mixin，它们在使用单个Django对象或多个对象时提供有用的功能。

通用编辑视图中还包含mixin（[`FormView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-editing.html#django.views.generic.edit.FormView)，以及特定于模型的视图[`CreateView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-editing.html#django.views.generic.edit.CreateView)，[`UpdateView`](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-editing.html#django.views.generic.edit.UpdateView)和[`DeleteView `](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/generic-editing.html#django.views.generic.edit.DeleteView)），以及基于日期的通用视图。 这些内容包含在[mixin参考文档](https://yiyibooks.cn/__trs__/qy/django2/ref/class-based-views/mixins.html)中。

### DetailView

**使用单个Django对象**

为了显示一个对象的详细信息，我们通常需要做两件事情：查询对象然后利用合适的模板和包含该对象的Context 生成TemplateResponse。

为了获得对象，DetailView 依赖SingleObjectMixin，它提供一个get_object() 方法，这个方法基于请求的URL 获取对象（它查找URLconf 中声明的pk 和slug 关键字参数，然后从视图的model 属性或queryset 属性查询对象）。 SingleObjectMixin 还覆盖`get_context_data()`，这个方法在Django 所有的内建的基于类的视图中都有用到，用来给模板的渲染提供Context 数据。

然后，为了生成TemplateResponse，DetailView 使用SingleObjectTemplateResponseMixin，它扩展自TemplateResponseMixin并覆盖上文讨论过的`get_template_names()`。 实际上，它提供比较复杂的选项集合，但是大部分人用到的主要的一个是` <app_label>/<model_name>_detail.html`。` _detail `部分可以通过设置子类的template_name_suffix 来改变。 （例如，generic edit views 使用`_form` 来创建和更新视图，用`_confirm_delete` 来删除视图）。

### ListView

**使用许多Django对象**

显示对象的列表和上面的步骤大体相同：我们需要一个对象的列表（可能是分页形式的），这通常是一个QuerySet，然后我们需要利用合适的模板和对象列表生成一个TemplateResponse。

为了获取对象，ListView 使用MultipleObjectMixin，它提供`get_queryset() `和`paginate_queryset()` 两种方法。 与SingleObjectMixin 不同，不需要根据URL 中关键字参数来获得查询集，默认将使用视图类的queryset 或model 属性。 通常需要覆盖`get_queryset() `以动态获取不同的对象，例如根据当前的用户或排除打算在将来提交的博客。

MultipleObjectMixin 还覆盖get_context_data() 以包含合适的Context 变量用于分页（如果禁止分页，则提供一些假的）。 这个方法依赖传递给它的关键字参数object_list，ListView 会负责准备好这个参数。

为了制作一个TemplateResponse，ListView然后使用MultipleObjectTemplateResponseMixin;与上面的SingleObjectTemplateResponseMixin一样，此方法将覆盖`get_template_names()`来提供一系列选项，其中最常用的是`<app_label> / <model_name> _list.html`，而`_list`部分再次取自template_name_suffix属性。（基于日期的通用视图使用诸如`_archive`，`_archive_year`之类的后缀，以对不同的基于日期的专用列表视图使用不同的模板。）

## 使用Django的基于类的视图混合
既然我们已经看到Django 通用的基于类的视图时如何使用Mixin，让我们在看看其它组合它们的方式。 当然，我们仍将它们与内建的基于类的视图或其它通用的基于类的视图组合，但是对于Django 提供的便利性你将解决一些更加罕见的问题。

> 警告
> 不是所有的Mixin 都可以一起使用，也不是所有的基于类的视图都可以与其它Mixin 一起使用。 在这里，我们提供几个做工作的例子；如果要汇集其他功能，则必须考虑属性和方法之间的交互，这些方法与您使用的不同类之间重叠，以及方法解析顺序将如何影响哪些版本的方法将以什么顺序调用。
>
> Django 的class-based views 和class-based view mixins 的文档将帮助你理解在不同的类和Mixin 之间那些属性和方法可能引起冲突。
>
> 如果有担心，通常最好退避并基于View 或TemplateView，或者可能的话加上SingleObjectMixin 和 MultipleObjectMixin。 虽然你可能最终会编写更多的代码，但是对于后来的人更容易理解，而且你自己也少了份担心。 （当然，您可以随时参与Django实施通用的基于类的视图，以启发如何解决问题。）

### 使用SingleObjectMixin与View 

如果你想编写一个简单的基于类的视图，它只响应post()，我们将子类化View 并在子类中只编写一个POST 方法。 但是，如果我们想处理一个由URL 标识的特定对象，我们将需要SingleObjectMixin 提供的功能。

我们将使用在[基于通用类的视图简介](https://yiyibooks.cn/__trs__/qy/django2/topics/class-based-views/generic-display.html)中用到的Author 模型做演示。
```python
# views.py
from django.http import HttpResponseForbidden, HttpResponseRedirect
from django.urls import reverse
from django.views import View
from django.views.generic.detail import SingleObjectMixin
from books.models import Author

class RecordInterest(SingleObjectMixin, View):
    """Records the current user's interest in an author."""
    model = Author

    def post(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return HttpResponseForbidden()
    
        # Look up the author we're interested in.
        self.object = self.get_object()
        # Actually record interest somehow here!
    
        return HttpResponseRedirect(reverse('author-detail', kwargs={'pk': self.object.pk}))
```
实际应用中，你的对象可能以键-值的方式保存而不是保存在关系数据库中，所以我们不考虑这点。 使用SingleObjectMixin 的视图唯一需要担心的是在哪里查询我们感兴趣的Author，而它会用一个简单的self.get_object() 调用实现。 其它的所有事情都有该Mixin 帮我们处理。

我们可以将它这样放入URL 中，非常简单：

```python
# urls.py
from django.conf.urls import url
from books.views import RecordInterest

urlpatterns = [
    #...
    url(r'^author/(?P<pk>[0-9]+)/interest/$', RecordInterest.as_view(), name='author-interest'),
]
```
注意Author 命名组，`get_object()` 将用它来查询pk 实例。 你还可以使用slug，或者SingleObjectMixin 的其它功能。

### 使用SingleObjectMixin与ListView 
ListView 提供内建的分页，但是可能你分页的列表中每个对象都与另外一个对象（通过一个外键）关联。 在我们的Publishing 例子中，你可能想通过一个特定的Publisher 分页所有的Book。

一种方法是组合ListView 和SingleObjectMixin，这样分页的Book 列表的查询集能够与找到的单个Publisher 对象关联。 为了实现这点，我们需要两个不同的查询集：

- Book查询使用ListView
由于我们可以访问我们要列出的图书的Publisher，我们只需覆盖get_queryset()，并使用Publisher的[反向外键管理员](https://yiyibooks.cn/__trs__/qy/django2/topics/db/queries.html#backwards-related-objects)。
-  Publisher在get_object()中使用的查询器
我们将依靠get_object()的默认实现来获取正确的Publisher对象。 但是，我们需要明确地传递一个queryset参数，否则默认的get_object()实现将调用我们覆盖的get_queryset()返回Book对象，而不是Publisher。

> 注
我们必须仔细考虑get_context_data()。 因为SingleObjectMixin 和ListView 都会将Context 数据的context_object_name 下，我们必须显式确保Publisher 位于Context 数据中。 ListView 将为我们添加合适的page_obj 和 paginator ，只要我们记住调用super()。

现在，我们可以编写一个新的PublisherDetail：

```python
from django.views.generic import ListView
from django.views.generic.detail import SingleObjectMixin
from books.models import Publisher

class PublisherDetail(SingleObjectMixin, ListView):
    paginate_by = 2
    template_name = "books/publisher_detail.html"

    def get(self, request, *args, **kwargs):
        self.object = self.get_object(queryset=Publisher.objects.all())
        return super(PublisherDetail, self).get(request, *args, **kwargs)
    
    def get_context_data(self, **kwargs):
        context = super(PublisherDetail, self).get_context_data(**kwargs)
        context['publisher'] = self.object
        return context
    
    def get_queryset(self):
        return self.object.book_set.all()
```
注意我们 在 get_queryset()方法里设置了get() ，这样我们就可以在后面的 get_context_data() 和self.object方法里再次用到它。如果您未设置template_name，则该模板将默认为普通的ListView选项，在本例中为`“ books / book_list.html”`，因为它是一本书的列表；ListView对SingleObjectMixin一无所知，因此没有任何线索可以证明此视图与发布者有关。

paginate_by是每页显示几条数据的意思，这里设的比较小，是因为这样你就不用造一堆数据才能看到分页的效果了！ 下面是你想要的模板:

```html
{% extends "base.html" %}

{% block content %}
    <h2>Publisher {{ publisher.name }}</h2>

    <ol>
      {% for book in page_obj %}
        <li>{{ book.title }}</li>
      {% endfor %}
    </ol>

    <div class="pagination">
        <span class="step-links">
            {% if page_obj.has_previous %}
                <a href="?page={{ page_obj.previous_page_number }}">previous</a>
            {% endif %}

            <span class="current">
                Page {{ page_obj.number }} of {{ paginator.num_pages }}.
            </span>

            {% if page_obj.has_next %}
                <a href="?page={{ page_obj.next_page_number }}">next</a>
            {% endif %}
        </span>
    </div>
{% endblock %}
```
## 避免任何更复杂的

通常情况下你只在需要相关功能时才会使用 TemplateResponseMixin和SingleObjectMixin这两个类。 如上所示，只要加点儿小心，你甚至可以把SingleObjectMixin和ListView结合在一起来使用. 但是这么搞可能会让事情变得有点复杂，作为一个好的原则：

> 提示
你的视图扩展应该仅仅使用那些来自于同一组通用基类的view或者mixins。如: detail, list, editing 和 date. 例如：把 TemplateView (内建视图)和 MultipleObjectMixin (通用列表)整合在一起是极好的, 但是若想把SingleObjectMixin (generic detail) 和 MultipleObjectMixin (generic list)整合在一起就有麻烦啦！

为了说明当你试图变得更复杂时会发生什么，我们展示了一个例子，当有一个更简单的解决方案时，牺牲了可读性和可维护性。 首先，让我们看一下将DetailView与FormMixin结合起来的天真尝试，使我们能够POST一个Django Form与我们使用DetailView显示对象的URL相同。

### 使用FormMixin与DetailView 

想想我们之前合用 View 和SingleObjectMixin 的例子. 我们正在记录用户对特定作者的兴趣；现在说，我们想让他们留言说他们为什么喜欢他们。 同样的，我们假设这些数据并没有存放在关系数据库里，而是存在另外一个奥妙之地（其实这里不用关心具体存放到了哪里）。

要实现这一点，自然而然就要设计一个 Form，让用户把相关信息通过浏览器发送到Django后台。 另外，我们要巧用REST方法,这样我们就可以用相同的URL来显示作者和捕捉来自用户的消息了。 让我们重写 AuthorDetailView 来实现它。

我们将保持DetailView的GET处理，虽然我们必须在上下文数据中添加一个Form，以便我们可以渲染它模板。 我们还想从FormMixin中提取表单处理，并写一些代码，以便在POST上适当地调用表单。

> 注
我们使用FormMixin并实现post()，而不是尝试将DetailView与FormView 结合(FormView已经提供了get()），因为这两个视图都实现了post()，事情会变得更加混乱。

我们的新AuthorDetail看起来像这样：
```python
# CAUTION: you almost certainly do not want to do this.
# It is provided as part of a discussion of problems you can
# run into when combining different generic class-based view
# functionality that is not designed to be used together.

from django import forms
from django.http import HttpResponseForbidden
from django.urls import reverse
from django.views.generic import DetailView
from django.views.generic.edit import FormMixin
from books.models import Author

class AuthorInterestForm(forms.Form):
    message = forms.CharField()

class AuthorDetail(FormMixin, DetailView):
    model = Author
    form_class = AuthorInterestForm

    def get_success_url(self):
        return reverse('author-detail', kwargs={'pk': self.object.pk})

    def get_context_data(self, **kwargs):
        context = super(AuthorDetail, self).get_context_data(**kwargs)
        context['form'] = self.get_form()
        return context

    def post(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return HttpResponseForbidden()
        self.object = self.get_object()
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form)
        else:
            return self.form_invalid(form)

    def form_valid(self, form):
        # Here, we would record the user's interest using the message
        # passed in form.cleaned_data['message']
        return super(AuthorDetail, self).form_valid(form)
```
form_valid()只是提供重定向的地方，它在get_success_url()的默认实现中使用。 如上所述，我们必须提供我们自己的get_context_data()，并覆盖post()，以使Form在上下文数据中可用。

### 优化方案

很明显，FormMixin和DetailView之间的微妙交互的数量已经在测试我们管理事物的能力。 你不太可能会去想自己写这种类的。

在这种情况下，只需自己写post()方法，保持DetailView作为唯一的通用功能，虽然写Form处理代码涉及很多重复。

或者，仍然比上述方法更容易具有用于处理表单的单独视图，这可以使用FormView与DetailView不同的地方。

### 其他方案

我们真正想在这里做的是使用来自同一个URL的两个不同的基于类的视图。 那么为什么不这样做呢？ 我们在这里有一个非常清楚的划分：POST请求应该获得DetailView（将Form添加到上下文数据），GET请求应该获得FormView。 让我们先设置这些视图。

AuthorDisplay视图与我们首次引入AuthorDetail时几乎相同。我们必须编写我们自己的`get_context_data()`，以使AuthorInterestForm可用于模板。为了清楚起见，我们将跳过之前的`get_object()`覆盖：

``` python
from django.views.generic import DetailView
from django import forms
from books.models import Author

class AuthorInterestForm(forms.Form):
    message = forms.CharField()

class AuthorDisplay(DetailView):
    model = Author

    def get_context_data(self, **kwargs):
        context = super(AuthorDisplay, self).get_context_data(**kwargs)
        context['form'] = AuthorInterestForm()
        return context
```
template_name是一个简单的 FormView, 但是我们不得不把SingleObjectMixin引入进来，这样我们才能定位我们评论的作者，并且我们还要记得设置AuthorDisplay来确保form出错时使用 GET会渲染到 AuthorInterest相同的模板 :
```python
from django.urls import reverse
from django.http import HttpResponseForbidden
from django.views.generic import FormView
from django.views.generic.detail import SingleObjectMixin

class AuthorInterest(SingleObjectMixin, FormView):
    template_name = 'books/author_detail.html'
    form_class = AuthorInterestForm
    model = Author

    def post(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return HttpResponseForbidden()
        self.object = self.get_object()
        return super(AuthorInterest, self).post(request, *args, **kwargs)

    def get_success_url(self):
        return reverse('author-detail', kwargs={'pk': self.object.pk})
```
最后，我们将这个在一个新的AuthorDetail视图中。 我们已经知道，在基于类的视图上调用as_view()会让我们看起来像一个基于函数的视图，所以我们可以在两个子视图之间选择。

您当然可以以与在URLconf中相同的方式将关键字参数传递给as_view()，例如，如果您希望AuthorInterest行为也出现在另一个网址但使用不同的模板：
```python
from django.views import View

class AuthorDetail(View):

    def get(self, request, *args, **kwargs):
        view = AuthorDisplay.as_view()
        return view(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        view = AuthorInterest.as_view()
        return view(request, *args, **kwargs)
```
此方法也可以与任何其他基于类的视图或您自己的基于类的视图直接从View或TemplateView继承，因为它独尽可能的保持不同的视图单。

## 返回HTML以外的内容 

当基于类的观点发光时，当你想做同样的事情很多次。 假设你正在编写API，每个视图应该返回JSON 而不是渲染后的HTML。

我们可以创建一个Mixin 类来处理JSON 的转换，并将它用于所有的视图。

例如，一个简单的JSON Mixin 可能像这样：
```python
from django.http import JsonResponse

class JSONResponseMixin(object):
    """
    A mixin that can be used to render a JSON response.
    """
    def render_to_json_response(self, context, **response_kwargs):
        """
        Returns a JSON response, transforming 'context' to make the payload.
        """
        return JsonResponse(
            self.get_data(context),
            **response_kwargs
        )

    def get_data(self, context):
        """
        Returns an object that will be serialized as JSON by json.dumps().
        """
        # Note: This is *EXTREMELY* naive; in reality, you'll need
        # to do much more complex handling to ensure that arbitrary
        # objects -- such as Django model instances or querysets
        # -- can be serialized as JSON.
        return context
```
> 注
查看Serializing Django objects 的文档，其中有如何正确转换Django 模型和查询集到JSON 的更多信息。

该Mixin 提供一个render_to_json_response() 方法，它与 render_to_response() 的参数相同。 要使用它，我们只需要将它与render_to_response() 组合，并覆盖render_to_json_response() 来调用TemplateView：
```python
from django.views.generic import TemplateView

class JSONView(JSONResponseMixin, TemplateView):
    def render_to_response(self, context, **response_kwargs):
        return self.render_to_json_response(context, **response_kwargs)
```
同样地，我们可以将我们的Mixin 与某个通用的视图一起使用。 我们可以实现自己的DetailView 版本，将JSONResponseMixin 和django.views.generic.detail.BaseDetailView 组合– (the DetailView before template rendering behavior has been mixed in):
```python
from django.views.generic.detail import BaseDetailView

class JSONDetailView(JSONResponseMixin, BaseDetailView):
    def render_to_response(self, context, **response_kwargs):
        return self.render_to_json_response(context, **response_kwargs)
```
这个视图可以和其它DetailView 一样使用，它们的行为完全相同 —— 除了响应的格式之外。

如果你想更进一步，你可以组合DetailView 的子类，它根据HTTP 请求的某个属性既能够返回HTML 又能够返回JSON 内容，例如查询参数或HTTP 头部。 这只需将JSONResponseMixin 和SingleObjectTemplateResponseMixin 组合，并覆盖render_to_response() 的实现以根据用户请求的响应类型进行正确的渲染：
```python
from django.views.generic.detail import SingleObjectTemplateResponseMixin

class HybridDetailView(JSONResponseMixin, SingleObjectTemplateResponseMixin, BaseDetailView):
    def render_to_response(self, context):
        # Look for a 'format=json' GET argument
        if self.request.GET.get('format') == 'json':
            return self.render_to_json_response(context)
        else:
            return super(HybridDetailView, self).render_to_response(context)
```
由于Python 解析方法重载的方式，`super(HybridDetailView, self).render_to_response(context)` 调用将以调用 TemplateResponseMixin 的`render_to_response()` 实现结束。
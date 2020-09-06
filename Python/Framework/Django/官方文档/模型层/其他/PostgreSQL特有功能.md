# `django.contrib.postgres`

PostgreSQL具有许多Django支持的其他数据库不共享的功能。该可选模块包含许多PostgreSQL特定数据类型的模型字段和表单字段。

> 注意
>
> Django是并且将继续是一个与数据库无关的Web框架。我们鼓励那些为Django社区编写可重用应用程序的人在可行的情况下编写与数据库无关的代码。但是，我们认识到使用Django编写的现实世界项目不必与数据库无关。实际上，一旦项目达到给定的大小，更改基础数据存储区已经是一个巨大的挑战，并且可能需要以某种方式更改代码库来处理数据存储区之间的差异。
>
> Django提供了对仅适用于PostgreSQL的许多数据类型的支持。没有根本的理由（例如）不存在contrib.mysql模块，只是PostgreSQL具有受支持数据库中功能最丰富的功能集，因此其用户受益最多。

- PostgreSQL specific aggregation functions
  - [General-purpose aggregation functions](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/aggregates.html#general-purpose-aggregation-functions)
  - [Aggregate functions for statistics](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/aggregates.html#aggregate-functions-for-statistics)
  - [Usage examples](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/aggregates.html#usage-examples)
- PostgreSQL specific model fields
  - [Indexing these fields](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/fields.html#indexing-these-fields)
  - [`ArrayField`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/fields.html#arrayfield)
  - [`CIText` fields](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/fields.html#citext-fields)
  - [`HStoreField`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/fields.html#hstorefield)
  - [`JSONField`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/fields.html#jsonfield)
  - [Range Fields](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/fields.html#range-fields)
- PostgreSQL specific form fields and widgets
  - [Fields](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/forms.html#fields)
  - [Widgets](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/forms.html#widgets)
- PostgreSQL specific database functions
  - [`RandomUUID`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/functions.html#randomuuid)
  - [`TransactionNow`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/functions.html#transactionnow)
- PostgreSQL specific model indexes
  - [`BrinIndex`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/indexes.html#brinindex)
  - [`GinIndex`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/indexes.html#ginindex)
  - [`GistIndex`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/indexes.html#gistindex)
- PostgreSQL specific lookups
  - [Trigram similarity](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/lookups.html#trigram-similarity)
  - [`Unaccent`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/lookups.html#unaccent)
- Database migration operations
  - [Creating extension using migrations](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/operations.html#creating-extension-using-migrations)
  - [`CreateExtension`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/operations.html#createextension)
  - [`BtreeGinExtension`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/operations.html#btreeginextension)
  - [`BtreeGistExtension`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/operations.html#btreegistextension)
  - [`CITextExtension`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/operations.html#citextextension)
  - [`CryptoExtension`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/operations.html#cryptoextension)
  - [`HStoreExtension`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/operations.html#hstoreextension)
  - [`TrigramExtension`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/operations.html#trigramextension)
  - [`UnaccentExtension`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/operations.html#unaccentextension)
- Full text search
  - [The `search` lookup](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/search.html#the-search-lookup)
  - [`SearchVector`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/search.html#searchvector)
  - [`SearchQuery`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/search.html#searchquery)
  - [`SearchRank`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/search.html#searchrank)
  - [Changing the search configuration](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/search.html#changing-the-search-configuration)
  - [Weighting queries](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/search.html#weighting-queries)
  - [Performance](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/search.html#performance)
  - [Trigram similarity](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/search.html#trigram-similarity)
- Validators
  - [`KeysValidator`](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/validators.html#keysvalidator)
  - [Range validators](https://yiyibooks.cn/__trs__/qy/django2/ref/contrib/postgres/validators.html#range-validators)
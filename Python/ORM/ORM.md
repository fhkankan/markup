# orm

## 概述

[参考](https://github.com/encode/orm)

orm软件包是用于Python的异步ORM，并支持Postgres，MySQL和SQLite。

ORM内置有：
```
SQLAlchemy核心用于查询构建。
跨数据库异步支持的databases。
用于数据验证的typesystem。
```
由于ORM基于SQLAlchemy核心构建，因此您可以使用Alembic提供数据库迁移。

## 使用

```python
import databases
import orm
import sqlalchemy

database = databases.Database("sqlite:///db.sqlite")
metadata = sqlalchemy.MetaData()


class Note(orm.Model):
    __tablename__ = "notes"
    __database__ = database
    __metadata__ = metadata

    id = orm.Integer(primary_key=True)
    text = orm.String(max_length=100)
    completed = orm.Boolean(default=False)

# Create the database
engine = sqlalchemy.create_engine(str(database.url))
metadata.create_all(engine)

# .create()
await Note.objects.create(text="Buy the groceries.", completed=False)
await Note.objects.create(text="Call Mum.", completed=True)
await Note.objects.create(text="Send invoices.", completed=True)

# .all()
notes = await Note.objects.all()

# .filter()
notes = await Note.objects.filter(completed=True).all()

# exact, iexact, contains, icontains, lt, lte, gt, gte, in
notes = await Note.objects.filter(text__icontains="mum").all()

# .get()
note = await Note.objects.get(id=1)

# .update()
await note.update(completed=True)

# .delete()
await note.delete()

# 'pk' always refers to the primary key
note = await Note.objects.get(pk=2)
note.pk  # 2
```

外键查询

```python
import databases
import orm
import sqlalchemy

database = databases.Database("sqlite:///db.sqlite")
metadata = sqlalchemy.MetaData()


class Album(orm.Model):
    __tablename__ = "album"
    __metadata__ = metadata
    __database__ = database

    id = orm.Integer(primary_key=True)
    name = orm.String(max_length=100)


class Track(orm.Model):
    __tablename__ = "track"
    __metadata__ = metadata
    __database__ = database

    id = orm.Integer(primary_key=True)
    album = orm.ForeignKey(Album)
    title = orm.String(max_length=100)
    position = orm.Integer()


# Create some records to work with.
malibu = await Album.objects.create(name="Malibu")
await Track.objects.create(album=malibu, title="The Bird", position=1)
await Track.objects.create(album=malibu, title="Heart don't stand a chance", position=2)
await Track.objects.create(album=malibu, title="The Waters", position=3)

fantasies = await Album.objects.create(name="Fantasies")
await Track.objects.create(album=fantasies, title="Help I'm Alive", position=1)
await Track.objects.create(album=fantasies, title="Sick Muse", position=2)


# Fetch an instance, without loading a foreign key relationship on it.
track = await Track.objects.get(title="The Bird")

# We have an album instance, but it only has the primary key populated
print(track.album)       # Album(id=1) [sparse]
print(track.album.pk)    # 1
print(track.album.name)  # Raises AttributeError

# Load the relationship from the database
await track.album.load()
assert track.album.name == "Malibu"

# This time, fetch an instance, loading the foreign key relationship.
track = await Track.objects.select_related("album").get(title="The Bird")
assert track.album.name == "Malibu"

# Fetch instances, with a filter across an FK relationship.
tracks = Track.objects.filter(album__name="Fantasies")
assert len(tracks) == 2

# Fetch instances, with a filter and operator across an FK relationship.
tracks = Track.objects.filter(album__name__iexact="fantasies")
assert len(tracks) == 2

# Limit a query
tracks = await Track.objects.limit(1).all()
assert len(tracks) == 1
```

支持以下字段类型

```
primary_key
allow_null
default
index
unique
```

除非设置以下任何一项，否则所有字段都是必填字段：

```
allow_null-创建可为空的列。将默认设置为无。
allow_blank-允许空字符串进行验证。将默认设置为“”。
default-为该字段设置默认值。
```

支持以下列类型

```
orm.String(max_length)
orm.Text()
orm.Boolean()
orm.Integer()
orm.Float()
orm.Date()
orm.Time()
orm.DateTime()
orm.JSON()
```


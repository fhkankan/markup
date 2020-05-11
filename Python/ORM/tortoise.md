# Tortoise

[参考](https://github.com/tortoise/tortoise-orm)

[文档](https://tortoise-orm.readthedocs.io/en/latest/)

## 概述

Tortoise ORM是受Django启发的易于使用的异步ORM（对象关系映射器）。

安装

```
pip install tortoise-orm
```

数据库驱动

```
pip install asyncpg  # PostgreSQL
pip install aiosqlite  # sqlite
pip install aiomysql  # mysql
```

## 使用

### 创建表

建立模型

```python
from tortoise.models import Model
from tortoise import fields

class Tournament(Model):
    id = fields.IntField(pk=True)
    name = fields.TextField()

    def __str__(self):
        return self.name


class Event(Model):
    id = fields.IntField(pk=True)
    name = fields.TextField()
    tournament = fields.ForeignKeyField('models.Tournament', related_name='events')
    participants = fields.ManyToManyField('models.Team', related_name='events', through='event_team')

    def __str__(self):
        return self.name


class Team(Model):
    id = fields.IntField(pk=True)
    name = fields.TextField()

    def __str__(self):
        return self.name
```

初始化

```python
from tortoise import Tortoise

async def init():
    # Here we connect to a SQLite DB file.
    # also specify the app name of "models"
    # which contain models from "app.models"
    await Tortoise.init(
        db_url='sqlite://db.sqlite3',
        modules={'models': ['app.models']}
    )
    # Generate the schema
    await Tortoise.generate_schemas()
```

### 使用表

```python
# Create instance by save
tournament = Tournament(name='New Tournament')
await tournament.save()

# Or by .create()
await Event.create(name='Without participants', tournament=tournament)
event = await Event.create(name='Test', tournament=tournament)
participants = []
for i in range(2):
    team = await Team.create(name='Team {}'.format(i + 1))
    participants.append(team)

# M2M Relationship management is quite straightforward
# (also look for methods .remove(...) and .clear())
await event.participants.add(*participants)

# You can query related entity just with async for
async for team in event.participants:
    pass

# After making related query you can iterate with regular for,
# which can be extremely convenient for using with other packages,
# for example some kind of serializers with nested support
for team in event.participants:
    pass


# Or you can make preemptive call to fetch related objects
selected_events = await Event.filter(
    participants=participants[0].id
).prefetch_related('participants', 'tournament')

# Tortoise supports variable depth of prefetching related entities
# This will fetch all events for team and in those events tournaments will be prefetched
await Team.all().prefetch_related('events__tournament')

# You can filter and order by related models too
await Tournament.filter(
    events__name__in=['Test', 'Prod']
).order_by('-events__participants__name').distinct()
```


# sys

This module provides access to some objects used or maintained by the interpreter and to functions that interact strongly with the interpreter.

## Dynamic objects

```

Dynamic objects:
    argv -- command line arguments; argv[0] is the script pathname if known
    path -- module search path; path[0] is the script directory, else ''
    modules -- dictionary of loaded modules

    displayhook -- called to show results in an interactive session
    excepthook -- called to handle any uncaught exception other than SystemExit
      To customize printing in an interactive session or to install a custom
      top-level exception handler, assign other functions to replace these.

    exitfunc -- if sys.exitfunc exists, this routine is called when Python exits
      Assigning to sys.exitfunc is deprecated; use the atexit module instead.

    stdin -- standard input file object; used by raw_input() and input()
    stdout -- standard output file object; used by the print statement
    stderr -- standard error object; used for error messages
      By assigning other file objects (or objects that behave like files)
      to these, it is possible to redirect all of the interpreter's I/O.

    last_type -- type of last uncaught exception
    last_value -- value of last uncaught exception
    last_traceback -- traceback of last uncaught exception
      These three are only available in an interactive session after a
      traceback has been printed.

    exc_type -- type of exception currently being handled
    exc_value -- value of exception currently being handled
    exc_traceback -- traceback of exception currently being handled
      The function exc_info() should be used instead of these three,
      because it is thread-safe.
```

## Static objects

```
Static objects:

    float_info -- a dict with information about the float inplementation.
    long_info -- a struct sequence with information about the long implementation.
    maxint -- the largest supported integer (the smallest is -maxint-1)
    maxsize -- the largest supported length of containers.
    maxunicode -- the largest supported character
    builtin_module_names -- tuple of module names built into this interpreter
    version -- the version of this interpreter as a string
    version_info -- version information as a named tuple
    hexversion -- version information encoded as a single integer
    copyright -- copyright notice pertaining to this interpreter
    platform -- platform identifier
    executable -- absolute path of the executable binary of the Python interpreter
    prefix -- prefix used to find the Python library
    exec_prefix -- prefix used to find the machine-specific Python library
    float_repr_style -- string indicating the style of repr() output for floats
    __stdin__ -- the original stdin; don't touch!
    __stdout__ -- the original stdout; don't touch!
    __stderr__ -- the original stderr; don't touch!
    __displayhook__ -- the original displayhook; don't touch!
    __excepthook__ -- the original excepthook; don't touch!
```

## function

```
Functions:
    displayhook() -- print an object to the screen, and save it in __builtin__._
    excepthook() -- print an exception and its traceback to sys.stderr
    exc_info() -- return thread-safe information about the current exception
    exc_clear() -- clear the exception state for the current thread
    exit() -- exit the interpreter by raising SystemExit
    getdlopenflags() -- returns flags to be used for dlopen() calls
    getprofile() -- get the global profiling function
    getrefcount() -- return the reference count for an object (plus one :-)
    getrecursionlimit() -- return the max recursion depth for the interpreter
    getsizeof() -- return the size of an object in bytes
    gettrace() -- get the global debug tracing function
    setcheckinterval() -- control how often the interpreter checks for events
    setdlopenflags() -- set the flags to be used for dlopen() calls
    setprofile() -- set the global profiling function
    setrecursionlimit() -- set the max recursion depth for the interpreter
    settrace() -- set the global debug tracing function
```

## demo

```
sys.vesion
# 获取解释器的版本信息

sys.path
# 获取模块的搜索路径，初始化时使用PYTHONPATH环境变量的值

sys.platform
# 获取操作系统平台的名称

sys.maxint
# 最大的int值

sys.maxunicode
# 最大的Unicode值

sys.stdin
# 获取信息到shell程序中

sys.stdout
# 向shell程序输出信息

sys.exit()
# 退出shell程序

sys._getframe(0)
# 获取当前栈信息

sys._getframe(0).f_code.co_filename
# 当前文件名

sys._getframe(0).f_code.co_name
# 当前函数名

sys._getframe(0).f_lineno
# 当前行号

sys._getframe().f_code.co_filename.split('/')[-1].split('.')[0]
# 获取文件名

sys.argv
# 接收命令行参数
# argv[0]:程序本身的名字
# argv[1:]:用户在命令行输入的参数
```


---
name: "\U0001F41B Bug report"
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
Please provide a clear and concise description of what the bug is.

If relevant, add a minimal example so that we can reproduce the error by running the code. It is very important for the snippet to be as succinct (minimal) as possible, so please take time to trim down any irrelevant code to help us debug efficiently. Your example should be fully self-contained and not rely on any artifact that should be downloaded. For example:

```python
# All necessary imports at the beginning
import afnio

# A succinct reproducing example trimmed down to the essential parts:
import afnio
import afnio.tellurio as te

te.login(api_key="TELLURIO_API_KEY")
run = te.init("user_handle", "My Project")

var = afnio.Variable(data="my data", role="variable role")
```

If the code is too long (hopefully, it isn't), feel free to put it in a public gist and link it in the issue: (https://gist.github.com)[https://gist.github.com].

Please also paste or describe the results you observe instead of the expected results. If you observe an error, please paste the error message including the full traceback of the exception. It may be relevant to wrap error messages in ```triple quotes blocks```.

**Versions**
Please run the following and paste the output below.

```
???
```

**Additional context**
Add any other context about the problem here.

Thanks for contributing 🎉!

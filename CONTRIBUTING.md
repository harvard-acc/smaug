How to contribute to SMAUG
----------------------

### If you found a bug: ###

Please file a GitHub issue or add a comment to an existing bug if it was
previously reported. In the issue, please describe the bug and provide
instructions to reproduce it.

If you would like to submit a fix for a bug, please make sure you add a new
test that fails without your fix and passes afterwards to prevent future
regressions.

### If you want to add a new feature: ###

File a GitHub issue proposing your suggested feature and discuss it with the
SMAUG maintainers. This does not have to be very lengthy, but if you are new to
SMAUG, it will help you a lot in understanding the context, things to watch out for,
and why certain decisions were made or not made. Once your feature is approved,
implement it and send a pull request for review.

When writing code, please adhere to the coding rules below:

### Coding style ###

1. Whenever possible, adhere to the existing conventions in the code you are
   modifying. For example, if the existing code uses `camelCase` for function
   names, you should do so too.
2. Write a descriptive commit message. If this PR implements a new feature, the
   commit message should describe how it can be used, such as with example code
   for a new API or operator. The commit message It should link
   to any GitHub issue it fixes or addresses.
3. All new features and bug fixes should include new unit tests to verify the
   functionality and prevent this same bug from recurring in the future.
4. Run `clang-format` on your code to keep it tidy automatically. From any
   directory inside the SMAUG repo, run `clang-format -i -style=file
   filename.cpp`. You can also configure your text editor to run this
   automatically every time you save the file.

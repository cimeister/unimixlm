package runner 

import (
  "fmt"
  "strings"
  "os"
  "os/exec"
  "syscall"
)

type Executor struct {}

type Context struct {
  BaseDir string
  Runner string
  Tool string
  Includes []string
  Args []string
}

func Execute(c Context) int {
  return execute(c.BaseDir, c.Runner, c.Tool, c.Includes, c.Args)
}

func execute(baseDir, runner, tool string, includes, args []string) int {
  cmd := newProcess(baseDir, runner, tool, includes, args)

  cmd.Stdout = os.Stdout
  cmd.Stderr = os.Stderr
  cmd.Stdin = os.Stdin

  err := cmd.Run()
  if err != nil {
    if exiterr, ok := err.(*exec.ExitError); ok {
      // Program has exited w/ exitcode != 0

      // Platform-dependent handling of this
      // see http://stackoverflow.com/questions/10385551/get-exit-code-go

      // This is supposed to work at least on Linux & Windows
      if status, ok := exiterr.Sys().(syscall.WaitStatus); ok {
        Log.Warn("Invocation ended w/ exit status ", status.ExitStatus())
        return status.ExitStatus()
      }
    }

    Log.Warn("Error invoking: ", err.Error())
  }

  Log.Info("Invocation ended sucessfully w/ exit status ", 0)
  return 0
}

func buildArgv(configs ConfigSource, baseDir, runner, tool string, includes, args []string) []string {
  useXp := configs.GetUse()
  Log.Debug("usexp  := ", useXp)

  if 0 == len(useXp) {
    panic("Cannot determine use_xp setting from " + configs.String())
  }

  runtime := configs.GetRuntime()
  Log.Debug("runt   := ", runtime)

  argv := []string {
    "-C", "-q", 
    "-d", fmt.Sprintf("include_path=\".%[2]s%[1]s%[2]s%[2]s%[3]s\"",
      strings.Join(useXp, string(os.PathListSeparator)),
      string(os.PathListSeparator),
      strings.Join(includes, string(os.PathListSeparator)),
    ),
    "-d", "magic_quotes_gpc=0",
  }

  for key, value := range configs.GetArgs(runtime) {
    argv = append(argv, "-d", key + "=\"" + value + "\"")
  }

  var runnerPath string
  if runnerPath = configs.Locate(useXp, "tools/" + runner + ".php"); runnerPath != "" {
    Log.Debug("Detected runner of XP < 6.0")
    // noop
  } else if runnerPath = configs.Locate([]string { baseDir }, runner + "-main.php"); runnerPath != "" {
    Log.Debug("Detected runner of XP >= 6.0, adding encoding information.")
    argv = append(argv, "-d", "encoding=\"utf-7\"")
  } else {
    panic("Cannot find tool in " + strings.Join(useXp, string(os.PathListSeparator)))
  }

  argv = append(argv, runnerPath, tool)
  if 0 < len(args) {
    argv = append(argv, args...)
  }

  Log.Debug("runner := ", runnerPath)

  return argv
}

func newProcess(baseDir, runner, tool string, includes, args []string) *exec.Cmd {
  configs := NewCompositeConfigSource(
    NewEnvironmentConfigSource(),
    IniConfigSourceFromFile(".", "xp.ini"),
    IniConfigSourceFromFile(os.Getenv("HOME"), ".xp.ini"),
    IniConfigSourceFromFile(baseDir, "xp.ini"),
  )

  executable := configs.GetExecutable(configs.GetRuntime())
  Log.Debug("exec   := ", executable)

  argv := buildArgv(configs, baseDir, runner, tool, includes, args)
  Log.Debug("cmdline := ", argv)

  return exec.Command(executable, argv...)
}
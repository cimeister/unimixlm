package main

import (
	"testing"

	"github.com/urfave/cli"
)

func run(t *testing.T, arguments []string) (Result, error) {
	var (
		result Result
		err    error
	)
	err = NewApp(func(c *cli.Context) error {
		result, err = runAction(c)
		if err != nil {
			return err
		}

		return nil
	}).Run(arguments)
	if err != nil {
		return Result{}, err
	}
	return result, nil
}

func TestPresentationError(t *testing.T) {
	arguments := []string{
		"sandbox",
		"--lang=c",
		"-c",
		"-s", "judge/src/A7/main.c",
		"-b", "judge/binary/A7/main",
		"-i", "judge/src/A7/input",
		"-o", "judge/src/A7/output",
	}
	r, err := run(t, arguments)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != "FE" {
		t.Log("result ", r)
		t.Fatal("Test Presentation Error Failed")
	}
}

func TestSegmentfault(t *testing.T) {
	arguments := []string{"sandbox",
		"--lang=c",
		"-c",
		"-s", "judge/src/A6/main.c",
		"-b", "judge/binary/A6/main",
		"-i", "judge/src/A6/input",
		"-o", "judge/src/A6/output",
	}
	r, err := run(t, arguments)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != "RE" {
		t.Logf("status: %s\n", r.Status)
		t.Fatal("Test Runtime Error Failed")
	}
}

func TestJudgeAPlusB(t *testing.T) {
	{
		arguments := []string{
			"sandbox",
			"--lang=c",
			"-c",
			"-s", "judge/src/A1/a+b.c",
			"-b", "judge/binary/A1/a+b",
			"-i", "judge/src/A1/input",
			"-o", "judge/src/A1/output",
		}
		r, err := run(t, arguments)
		if err != nil {
			t.Fatal(err)
		}
		if r.Status != "AC" {
			t.Fatal("wrong answer")
		}
	}
	{
		// Test command with no -c flag
		arguments := []string{
			"sandbox",
			"--lang=c",
			"-s", "judge/src/A1/a+b.c",
			"-b", "judge/binary/A1/a+b",
			"-i", "judge/src/A1/input",
			"-o", "judge/src/A1/output",
		}
		r, err := run(t, arguments)
		if err != nil {
			t.Fatal(err)
		}

		if r.Status != "AC" {
			t.Fatal("Run With compiled binary failed")
		}
	}

}

func TestNTimesAPlusB(t *testing.T) {
	arguments := []string{
		"sandbox",
		"--lang=c",
		"-c",
		"-s", "judge/src/A3/main.c",
		"-b", "judge/binary/A3/main",
		"-i", "judge/src/A3/input",
		"-o", "judge/src/A3/output",
	}
	r, err := run(t, arguments)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != "WA" || r.Nth != 2 {
		t.Fatalf("Test N times of test failed result: %#v\n", r)
	}
}

func TestTimeLimit(t *testing.T) {
	{
		arguments := []string{"sandbox",
			"--lang=c",
			"-c",
			"-s", "judge/src/A2/main.c",
			"-b", "judge/binary/A2/main",
		}
		r, err := run(t, arguments)
		if err != nil {
			t.Fatal(err)
		}
		if r.Status != "TL" {
			t.Fatal("Test busy loop out of time error")
		}
	}

	{
		arguments := []string{
			"sandbox",
			"--lang=c",
			"-c",
			"-s", "judge/src/A8/main.c",
			"-b", "judge/binary/A8/main",
			"-i", "judge/src/A8/input",
			"-o", "judge/src/A8/output",
		}
		r, err := run(t, arguments)
		if err != nil {
			t.Fatal(err)
		}
		if r.Status != "TL" {
			t.Fatal("Test out of time error")
		}

	}
}
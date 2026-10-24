using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Nono.Engine
{
    public class TaskCollection : IReadOnlyCollection<TaskLine>
    {
        private readonly IReadOnlyDictionary<Orientation, TaskLine[]> _tasks;

        private TaskCollection(IReadOnlyDictionary<Orientation, TaskLine[]> tasks, int count)
        {
            _tasks = tasks;
            Count = count;
        }

        public int Count { get; private set; }

        public TaskLine this[Orientation orienation, int position] => _tasks[orienation][position];

        public static TaskCollection Create(Nonogram nonogram)
        {
            var tasks = new Dictionary<Orientation, TaskLine[]>()
            {
                [Orientation.Column] = CreateTaskLines(nonogram.Columns, nonogram.RowsCount, Orientation.Column).ToArray(),
                [Orientation.Row] = CreateTaskLines(nonogram.Rows, nonogram.ColumnsCount, Orientation.Row).ToArray(),
            };

            return new TaskCollection(tasks, nonogram.Rows.Length + nonogram.Columns.Length);
        }

        private static IEnumerable<TaskLine> CreateTaskLines(int[][] tasksAxis, int length, Orientation orienation)
        {
            for (int i = 0; i < tasksAxis.Length; i++)
                yield return new TaskLine(tasksAxis[i], length, new LineIndex(orienation, i));
        }

        public IEnumerator<TaskLine> GetEnumerator()
            => _tasks.Values.SelectMany(x => x).GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
using System;

namespace Bardiche.JSONModels
{
    public class MangaResult
    {
        public int id;
        public string publishing_status;
        public string image_url_lge;
        public string title_japanese;
        public string type;
        public int total_chapters;
        public int total_volumes;
        public string start_date;
        public string end_date;
        public string average_score;
        //public string description;

        public override string ToString() =>
            "``Title: " + title_japanese + "``" +
            "\n``Type: " + type + "``" +
            "\n``Status: " + publishing_status + "``" +
            (total_chapters != 0 ? ("\n``Chapters: " + total_chapters + "``") : null) +
            (total_volumes != 0 ? ("\n``Volumes: " + total_volumes + "``") : null) +
            (!string.IsNullOrWhiteSpace(start_date) ? ("\n``Start Date: " + start_date.Substring(0, 10)  + "``") : null) +
            (!string.IsNullOrWhiteSpace(end_date) ? ("\n``End Date: " + end_date.Substring(0, 10) + "``") : null) +
            (Double.Parse(average_score) != 0 ? ("\n``Rating: " + average_score + "``") : null) +
            "\n``Link:`` http://anilist.co/manga/" + id +
            /*"\n``Synopsis:`` " + description.Substring(0, description.Length > 500 ? 500 : description.Length) + "..." +*/
            "\n" + image_url_lge;
    }
}
using Dapper;
using Dapper.Contrib.Extensions;
using DapperFantom.Entities;
using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Threading.Tasks;

namespace DapperFantom.Services
{
    public class ArticleService
    {
        private IDbConnection _dapperConnection;
        private ConnectionService _connectionService;

        public ArticleService(IConfiguration config)
        {
            this._connectionService = new ConnectionService(config);
            this._dapperConnection = this._connectionService.ForDapper();
        }

        public int Add(Article article)
        {
            // The Dapper table plural nomenclatures are a little bit annoying...
            var result = this._dapperConnection.Insert(article);

            if(result > 0)
            {
                return int.Parse(result.ToString());
            }
            return 0;
        }

        internal int GetTotalArticleCountByCategory(int categoryId)
        {
            //List<Article> articleLst = new List<Article>();
            //articleLst = GetArticlesByCategoryId(categoryId);
            //return articleLst.Count();

            var sql = $@"select count(CategoryId) from [Articles] where [CategoryId] = {categoryId}";
            return Convert.ToInt32(this._dapperConnection.ExecuteScalar(sql).ToString());
        }

        public List<Article> GetAllArticles()
        {
            List<Article> articleLst = new List<Article>();
            try
            {
                articleLst = this._dapperConnection.Query<Article>(@"select * from [Articles]").ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return articleLst;
        }

        public List<Article> GetArticlesByCategoryId(int categoryId, int page = 1)
        {
            List<Article> articleLst = new List<Article>();

            try
            {
                int articlesPerPage = 3;
                int offset = (page - 1) * articlesPerPage;
                var par = new DynamicParameters();
                par.Add("@CategoryId", categoryId);
                par.Add("@articlesPerPage", articlesPerPage);
                par.Add("@offset", offset);
                articleLst = this._dapperConnection.Query<Article>($@"select * from [Articles] where [Status] = 1 
                                and [CategoryId] = @CategoryId
                                order by [PublishedDate] desc
                                offset @offset rows
                                fetch next @articlesPerPage rows only", par).ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return articleLst;
        }

        public List<Article> GetArticlesBySerchTerm(string searchQuery)
        {
            List<Article> articleLst = new List<Article>();

            try
            {
                // Using Take() to only return 5 articles (I don't like it, but...)
                articleLst = this._dapperConnection.Query<Article>($@"select * from [Articles] where [Articles].[Status] = 1 
                                and [Articles].[Title] like @searchQuery
                                or [Articles].[Content] like @searchQuery
                                order by [Articles].[ArticleId] desc", new { searchQuery = "%" + searchQuery + "%" }).Take(5).ToList();
                //articleLst = this._dapperConnection.Query<Article>(@"select * from [Articles] where [Status] = 1 and CONTAINS([Title], '@term') or CONTAINS([Content], '@term') order by [ArticleId] desc", par).ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return articleLst;
        }

        public List<Article> GetArticlesPerPage(int page = 1)
        {
            List<Article> articleLst = new List<Article>();

            try
            {
                int articlesPerPage = 3;
                int offset = (page - 1) * articlesPerPage;
                var par = new DynamicParameters();
                par.Add("@articlesPerPage", articlesPerPage);
                par.Add("@offset", offset);

                var sql = $@"select * from [Articles]
                            inner join [Categorys] as cat on cat.[CategoryId] = [Articles].[CategoryId]
                            where [Homeview] = 1 and [Status] = 1 or [Status] = 1
                            order by [PublishedDate] desc
                            offset @offset rows
                            fetch next @articlesPerPage rows only";

                articleLst = this._dapperConnection.Query<Article, Category, Article>(sql, (article, category) => {
                    article.Category = category;
                    return article;
                }, 
                par, splitOn: "CategoryId").ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return articleLst;
        }

        public Article IncreaseHit(int id)
        {
            var sql = $@"update [Articles] set [Hit] = [Hit] + 1 where [ArticleId] = {id}";
            try
            {
                return this._dapperConnection.Query<Article>(sql).FirstOrDefault();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return null;
            }
           
        }

        public int CountTotalArticles()
        {
            var sql = $@"select count(ArticleId) from [Articles] where [Status] = 1";
            return Convert.ToInt32(this._dapperConnection.ExecuteScalar(sql).ToString());
        }

        public List<Article> GetHomeArticles()
        {
            List<Article> articleLst = new List<Article>();

            try
            {
                articleLst = this._dapperConnection.Query<Article>(@"select * from [Articles] where [Status] = 1 or [Slider] = 1 and [HomeView] = 1").ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return articleLst;
        }

        public List<Article> GetAllArticlesAlt()
        {
            List<Article> articleLst = new List<Article>();

            try
            {
                articleLst = this._dapperConnection.GetAll<Article>().ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return articleLst;
        }

        public Article GetArticleById(int id)
        {
            Article article = new Article();

            try
            {
                article = this._dapperConnection.Query<Article>($"select * from [Articles] where [Articles].[ArticleId] = {id}").FirstOrDefault();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return article;
        }

        public Article GetArticleByIdAlt(int id)
        {
            Article article = new Article();

            try
            {
                article = this._dapperConnection.Get<Article>(id);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return article;
        }


        public Article GetArticleByGuid(string guid)
        {
            Article article = new Article();

            try
            {
                var par = new DynamicParameters();
                par.Add("@guid", guid);
                article = this._dapperConnection.Query<Article>(@"select * from [Articles] where [Articles].[Guid] = @guid", par).FirstOrDefault();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return article;
        }

        public Article GetPrevArticleById(int id)
        {
            Article prevArticle = new Article();
            var par = new DynamicParameters();

            try
            {
                par.Add("@ArticleId", id);
                string sql = @"select [ArticleId], [Guid], [Title], [Image] from [Articles] 
                            where [ArticleId]<@ArticleId order by [ArticleId] desc
                            offset 0 rows
                            fetch next 1 rows only";
                prevArticle = this._dapperConnection.Query<Article>(sql, par).FirstOrDefault();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return prevArticle;
        }

        public Article GetNextArticleById(int id)
        {
            Article nextArticle = new Article();
            var par = new DynamicParameters();

            try
            {
                par.Add("@ArticleId", id);
                string sql = @"select top 1 [ArticleId], [Guid], [Title], [Image] from [Articles] 
                            where [ArticleId]>@ArticleId order by [ArticleId] asc";
                nextArticle = this._dapperConnection.Query<Article>(sql, par).FirstOrDefault();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return nextArticle;
        }

        public bool UpdateArticle(Article article)
        {
            // https://localhost:44377/Admin/Article
            Article origArt = GetArticleByIdAlt(article.ArticleId);

            article.ModifiedDate = DateTime.Now;
            article.Seen = 1;
            article.Guid = origArt.Guid;
            article.Image = origArt.Image;
            article.PublishedDate = origArt.PublishedDate;
            article.CreatedDate = origArt.CreatedDate;

            try
            {
                bool result = this._dapperConnection.Update<Article>(article);
                return result;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return false;
            }
        }

        public List<Article> GetArticlesByStatus(int status)
        {
            List<Article> articleLst = new List<Article>();

            try
            {
                var par = new DynamicParameters();
                par.Add("@Status", status);

                articleLst = this._dapperConnection.Query<Article>(@"select * from [Articles] where [Articles].[Status] = @Status", par).ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return articleLst;
        }

        public bool DeleteArticle(Article article)
        {
            try
            {
                bool result = this._dapperConnection.Delete(article);
                return result;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return false;
            }
        }
    }
}
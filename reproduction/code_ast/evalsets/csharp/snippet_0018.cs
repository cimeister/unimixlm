using LojaSapatos.Modelos;
using System;
using Xamarin.Forms;
using Xamarin.Forms.Xaml;
using SQLite;
using System.IO;
using LojaSapatos.Servicos;

namespace LojaSapatos
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();

            MainPage = new MainPage();

            //criarDados();
        }

        public void criarDados()
        {
            //Pegar a conexão com o banco local
            var db = DB.getConnection();

            Cliente cliente = new Cliente();

            cliente.Cpf = "999.999.999-99";
            cliente.Endereco.Bairro = "Vila Universitária";
            cliente.Endereco.Cep = "17012-631";
            cliente.Endereco.Cidade = "Bauru";
            cliente.Endereco.Estado = "SP";
            cliente.Endereco.Logradouro = "Rua Engenheiro Alpheu José Ribas Sampaio";
            cliente.Endereco.Numero = "2-25";
            cliente.Id = 123456;
            cliente.Nome = "<NAME>";

            db.Insert(cliente);  //insere o cliente no banco

            Loja loja = new Loja();

            loja.Id = 456789;
            loja.Cnpj = "6547891254";
            loja.Endereco.Logradouro = "Teste teste";

            db.Insert(loja);  //insere a loja no banco

            Venda venda = new Venda();

            venda.Loja = loja;
            venda.Cliente = cliente;
        }

        protected override void OnStart()
        {
        }

        protected override void OnSleep()
        {
        }

        protected override void OnResume()
        {
        }
    }
}
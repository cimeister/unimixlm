using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using cleanArchitecture.Core.Interfaces.Repositories;
using cleanArchitecture.UnitTests.MockObjects;
using cleanArchitecture.Web.Controllers;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Moq;
using Xunit;
using ProductAggregate = cleanArchitecture.Core.Entities.ProductAggregate;
using Messages = cleanArchitecture.Web.Messages;



namespace cleanArchitecture.UnitTests.Controllers
{
    public class ProductControllerTests
    {
        public ProductControllerTests()
        {
        }

        [Fact]
        public async Task GetProducts_ListAsync()
        {
            //Arrange
            var productRepository = new Mock<IAsyncRepository<ProductAggregate.Product>>();
            productRepository.Setup(repo => repo.ListAllAsync())
                .ReturnsAsync(Product.GetProducts())
                .Verifiable();

            var controlller = new ProductController(productRepository.Object);

            //Act
            var result = await controlller.GetAsync();

            //Assert
            var okObjectResult = result as OkObjectResult;


            var products = okObjectResult.Value as IEnumerable<Object>;

            productRepository.Verify();
            Assert.NotNull(okObjectResult);
            Assert.NotNull(products);
            Assert.NotEmpty(products);           
        }

        [Fact]
        public async Task GetProducts_Empty()
        {
            //Arrange
            var productRepository = new Mock<IAsyncRepository<ProductAggregate.Product>>();
            productRepository.Setup(repo => repo.ListAllAsync())
                .ReturnsAsync(Product.GetProducts_Empty())
                .Verifiable();

            var controlller = new ProductController(productRepository.Object);

            //Act
            var result = await controlller.GetAsync();
            var notFoundResult = result as NotFoundResult;

            //Assert
            Assert.NotNull(notFoundResult);   
            Assert.Equal(StatusCodes.Status404NotFound, notFoundResult.StatusCode);
        }

        [Fact]
        public async Task GetProductById_Exists()
        {
            //Arrange
            var productRepository = new Mock<IAsyncRepository<ProductAggregate.Product>>();
            productRepository.Setup(repo => repo.GetByIdAsync(It.IsAny<Guid>()))
                .ReturnsAsync(Product.GetProduct())
                .Verifiable();
            

            var controlller = new ProductController(productRepository.Object);

            //Act
            var result = await controlller.GetAsync(new Guid());

            //Assert
            var okObjectResult = result as OkObjectResult;


            var product = okObjectResult.Value as object;


            productRepository.Verify();
            Assert.NotNull(okObjectResult);
            Assert.NotNull(product);
        }

        [Fact]
        public async Task GetProductById_DoesNotExist()
        {
            //Arrange
            var productRepository = new Mock<IAsyncRepository<ProductAggregate.Product>>();
            productRepository.Setup(repo => repo.GetByIdAsync(It.IsAny<Guid>()))
                .ReturnsAsync(Product.GetProduct_Empty())
                .Verifiable();

            var controlller = new ProductController(productRepository.Object);

            //Act
            var result = await controlller.GetAsync(new Guid());
            var notFoundResult = result as NotFoundResult;

            //Assert
            productRepository.Verify();
            Assert.NotNull(notFoundResult);
            Assert.Equal(StatusCodes.Status404NotFound, notFoundResult.StatusCode);
        }

        [Fact]
        public async Task PostProduct()
        {
            //Arrange
            var productRepository = new Mock<IAsyncRepository<ProductAggregate.Product>>();
            productRepository.Setup(repo => repo.AddAsync(It.IsAny<ProductAggregate.Product>()))
                .Returns(
                (ProductAggregate.Product p) =>
                {
                    p.Id = new Guid();
                    return Task.FromResult(p);
                }).Verifiable();

            var controlller = new ProductController(productRepository.Object);

            //Act
            var productDTO = Product.GetProductDTO();
            var result = await controlller.PostAsync(productDTO);
            var createdResult = result as CreatedResult;

            var product = createdResult.Value as Messages.Product;


            //Assert
            productRepository.Verify();
            Assert.NotNull(createdResult);

            Assert.NotNull(product);

            Assert.NotNull(product.Id);
            Assert.Equal(productDTO.Name, product.Name);
            Assert.Equal(productDTO.Price, product.Price);
            Assert.Equal(productDTO.DeliveryPrice, product.DeliveryPrice);
            Assert.Equal(productDTO.Description, product.Description);

            Assert.Equal(StatusCodes.Status201Created, createdResult.StatusCode);
        }

        [Fact]
        public async Task UpdateProduct_Exists()
        {
            //Arrange
            var productRepository = new Mock<IAsyncRepository<ProductAggregate.Product>>();
            var product = Product.GetProduct();
            product.Id = new Guid();
            
            productRepository.Setup(repo => repo.GetByIdAsync(It.IsAny<Guid>()))
                .ReturnsAsync(product)
                .Verifiable();            

            productRepository.Setup(repo => repo.UpdateAsync(It.IsAny<ProductAggregate.Product>()))
                .Returns(
                (ProductAggregate.Product p) =>
                {
                    return Task.FromResult(p);
                }).Verifiable();

            var controlller = new ProductController(productRepository.Object);

            //Act
            var productDTO = Product.GetProductDTO(product);

            var result = await controlller.UpdateAsync(productDTO);

            var createdResult = result as CreatedResult;

            var productResponseDto = createdResult.Value as Messages.Product;


            //Assert
            productRepository.Verify();
            Assert.NotNull(createdResult);

            Assert.NotNull(productResponseDto);

            Assert.NotNull(product.Id);
            Assert.Equal(productDTO.Name, productResponseDto.Name);
            Assert.Equal(productDTO.Price, productResponseDto.Price);
            Assert.Equal(productDTO.DeliveryPrice, productResponseDto.DeliveryPrice);
            Assert.Equal(productDTO.Description, productResponseDto.Description);

            Assert.Equal(StatusCodes.Status201Created, createdResult.StatusCode);
        }

        [Fact]
        public async Task UpdateProduct_DoesNotExist()
        {
            //Arrange
            var productRepository = new Mock<IAsyncRepository<ProductAggregate.Product>>();

            productRepository.Setup(repo => repo.GetByIdAsync(It.IsAny<Guid>()))
                .ReturnsAsync(Product.GetProduct_Empty())
                .Verifiable();

            productRepository.Setup(repo => repo.UpdateAsync(It.IsAny<ProductAggregate.Product>()))
                .Returns(
                (ProductAggregate.Product p) =>
                {
                    return Task.FromResult(p);
                });

            var controlller = new ProductController(productRepository.Object);

            //Act
            var productDTO = Product.GetProductDTO();
            productDTO.Id = new Guid();

            var result = await controlller.UpdateAsync(productDTO);

            var notFoundResult = result as NotFoundResult;

            //Assert
            productRepository.Verify();
            productRepository.VerifyNoOtherCalls();


            //Verify
            Assert.NotNull(notFoundResult);
            Assert.Equal(StatusCodes.Status404NotFound, notFoundResult.StatusCode);
        }

        [Fact]
        public async Task Delete_Exists()
        {
            //Arrange
            var productRepository = new Mock<IAsyncRepository<ProductAggregate.Product>>();

            productRepository.Setup(repo => repo.GetByIdAsync(It.IsAny<Guid>()))
                .ReturnsAsync(Product.GetProduct())
                .Verifiable();

            productRepository.Setup(repo => repo.DeleteAsync(It.IsAny<ProductAggregate.Product>()))
                .Returns(Task.CompletedTask)
                .Verifiable();

            var controlller = new ProductController(productRepository.Object);

            //Act
            var result = await controlller.DeleteAsync(new Guid());

            //Assert            
            var acceptedResult = result as AcceptedResult;


            productRepository.Verify();

            Assert.NotNull(acceptedResult);
            Assert.Equal(StatusCodes.Status202Accepted, acceptedResult.StatusCode);
        }

        [Fact]
        public async Task Delete_DoesNotExist()
        {
            //Arrange
            var productRepository = new Mock<IAsyncRepository<ProductAggregate.Product>>();

            productRepository.Setup(repo => repo.GetByIdAsync(It.IsAny<Guid>()))
                .ReturnsAsync(Product.GetProduct_Empty())
                .Verifiable();

            productRepository.Setup(repo => repo.DeleteAsync(It.IsAny<ProductAggregate.Product>()))
                .Returns(Task.CompletedTask);
                

            var controlller = new ProductController(productRepository.Object);

            //Act
            var result = await controlller.DeleteAsync(new Guid());

            //Assert
            var notFoundResult = result as NotFoundResult;

            productRepository.Verify();
            productRepository.VerifyNoOtherCalls();
            Assert.NotNull(notFoundResult);

            Assert.Equal(StatusCodes.Status404NotFound, notFoundResult.StatusCode);
        }
    }
}
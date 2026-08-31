<?php

namespace FondOfSpryker\Glue\UnpaginatedCatalogSearchRestApi\Controller;

use Codeception\Test\Unit;
use FondOfSpryker\Glue\UnpaginatedCatalogSearchRestApi\Processor\Catalog\UnpaginatedCatalogSearchReaderInterface;
use FondOfSpryker\Glue\UnpaginatedCatalogSearchRestApi\UnpaginatedCatalogSearchRestApiFactory;
use Spryker\Glue\GlueApplication\Rest\JsonApi\RestResponseInterface;
use Spryker\Glue\GlueApplication\Rest\Request\Data\RestRequestInterface;

class UnpaginatedCatalogSearchResourceControllerTest extends Unit
{
    /**
     * @var \FondOfSpryker\Glue\UnpaginatedCatalogSearchRestApi\Controller\UnpaginatedCatalogSearchResourceController
     */
    protected $unpaginatedCatalogSearchResourceController;

    /**
     * @var \PHPUnit\Framework\MockObject\MockObject|\FondOfSpryker\Glue\UnpaginatedCatalogSearchRestApi\UnpaginatedCatalogSearchRestApiFactory
     */
    protected $unpaginatedCatalogSearchRestApiFactoryMock;

    /**
     * @var \PHPUnit\Framework\MockObject\MockObject|\Spryker\Glue\GlueApplication\Rest\Request\Data\RestRequestInterface
     */
    protected $restRequestInterfaceMock;

    /**
     * @var \PHPUnit\Framework\MockObject\MockObject|\FondOfSpryker\Glue\UnpaginatedCatalogSearchRestApi\Processor\Catalog\UnpaginatedCatalogSearchReaderInterface
     */
    protected $unpaginatedCatalogSearchReaderInterfaceMock;

    /**
     * @var \PHPUnit\Framework\MockObject\MockObject|\Spryker\Glue\GlueApplication\Rest\JsonApi\RestResponseInterface
     */
    protected $restResponseInterfaceMock;

    /**
     * @return void
     */
    protected function _before(): void
    {
        $this->unpaginatedCatalogSearchRestApiFactoryMock = $this->getMockBuilder(UnpaginatedCatalogSearchRestApiFactory::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->restRequestInterfaceMock = $this->getMockBuilder(RestRequestInterface::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->unpaginatedCatalogSearchReaderInterfaceMock = $this->getMockBuilder(UnpaginatedCatalogSearchReaderInterface::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->restResponseInterfaceMock = $this->getMockBuilder(RestResponseInterface::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->unpaginatedCatalogSearchResourceController = new class (
            $this->unpaginatedCatalogSearchRestApiFactoryMock
        ) extends UnpaginatedCatalogSearchResourceController {
            /**
             * @var \FondOfSpryker\Glue\UnpaginatedCatalogSearchRestApi\UnpaginatedCatalogSearchRestApiFactory
             */
            protected $unpaginatedCatalogSearchRestApiFactory;

            /**
             * @param \FondOfSpryker\Glue\UnpaginatedCatalogSearchRestApi\UnpaginatedCatalogSearchRestApiFactory $unpaginatedCatalogSearchRestApiFactory
             */
            public function __construct(UnpaginatedCatalogSearchRestApiFactory $unpaginatedCatalogSearchRestApiFactory)
            {
                $this->unpaginatedCatalogSearchRestApiFactory = $unpaginatedCatalogSearchRestApiFactory;
            }

            /**
             * @return \FondOfSpryker\Glue\UnpaginatedCatalogSearchRestApi\UnpaginatedCatalogSearchRestApiFactory
             */
            public function getFactory(): UnpaginatedCatalogSearchRestApiFactory
            {
                return $this->unpaginatedCatalogSearchRestApiFactory;
            }
        };
    }

    /**
     * @return void
     */
    public function testGetAction(): void
    {
        $this->unpaginatedCatalogSearchRestApiFactoryMock->expects($this->atLeastOnce())
            ->method('createUnpaginatedCatalogSearchReader')
            ->willReturn($this->unpaginatedCatalogSearchReaderInterfaceMock);

        $this->unpaginatedCatalogSearchReaderInterfaceMock->expects($this->atLeastOnce())
            ->method('unpaginatedCatalogSearch')
            ->with($this->restRequestInterfaceMock)
            ->willReturn($this->restResponseInterfaceMock);

        $this->assertInstanceOf(
            RestResponseInterface::class,
            $this->unpaginatedCatalogSearchResourceController->getAction(
                $this->restRequestInterfaceMock
            )
        );
    }
}
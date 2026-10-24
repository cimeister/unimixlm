<?php
/**
 * PHP version 7.1
 *
 * This source file is subject to the license that is bundled with this package in the file LICENSE.
 */

namespace Ads\UI\Web\Slim\Controllers;

use Ads\Application\CommandBus\Bus;
use Ads\CodeList\Ads\Ad;
use Ads\CodeList\Listings\DraftAd\DraftAdAction;
use Ads\CodeList\Listings\DraftAd\DraftAdInput;
use Ads\CodeList\Listings\DraftAd\DraftAdResponder;
use Ads\CodeList\Posters\Username;
use Ads\UI\Web\HTTP\ApiResponse;
use Ads\UI\Web\HTTP\HAL\ApiProblems\Problem;
use Ads\UI\Web\HTTP\HAL\ApiProblems\ProblemDetails;
use Ads\UI\Web\HTTP\HAL\Mappings\SlimUriBuilder;
use Ads\UI\Web\HTTP\HAL\Serializer;
use Psr\Http\Message\ResponseInterface;
use Slim\Http\Request;
use Slim\Router;

class DraftAdController extends ApiController implements DraftAdResponder
{
    public function __construct(Bus $bus, Router $router, DraftAdAction $action)
    {
        parent::__construct($bus, $router);
        $action->attach($this);
        $this->bus->addHandler($action, 'draftAd', DraftAdInput::class);
    }

    public function draftNewAd(Request $request): ResponseInterface
    {
        return $this->run($request, DraftAdInput::withValues($request->getParsedBody()));
    }

    public function respondToAdDrafted(Ad $ad): void
    {
        $uriBuilder = new SlimUriBuilder($this->router, $this->request);
        $serializer = Serializer::hal($uriBuilder);

        $this->response = ApiResponse::created(
            $uriBuilder->pathFor('draftAd', ['id' => $ad->id()]),
            $serializer->serializeItem($ad)
        );
    }

    public function respondToInvalidInput(DraftAdInput $input): void
    {
        $this->response = ApiResponse::unprocessableEntity(
            Problem::failedValidation($input->errors(), ProblemDetails::INVALID_DRAFT_INFORMATION)
        );
    }

    public function respondToUnknownPosterWith(Username $username): void
    {
        $this->response = ApiResponse::notFound(Problem::notFound(
            ['username' => "Cannot find user with username $username"],
            ProblemDetails::POSTER_NOT_FOUND
        ));
    }
}
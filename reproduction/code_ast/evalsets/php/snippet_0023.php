@inject('blocksHelper', 'App\Services\BlocksHelper')
<div class="header-title service">
    <div class="content">
        <div class="breadcrumbs" itemscope itemtype="https://schema.org/BreadcrumbList">
            @foreach ($model->getBreadcrumbs() as $item)
                @if ($item['url'])
                    <span itemprop="itemListElement" itemscope itemtype="https://schema.org/ListItem">
                            <a itemprop="item" href="{{ $item['url'] }}">
                                <span itemprop="name">{{ $item['name'] }}</span>
                            </a>
                        <meta itemprop="position" content="{{ $loop->index + 1 }}"/>
                    </span>
                    <span> / </span>
                @else
                    <span>
                        <span>
                            {{ $item['name'] }}
                        </span>
                    </span>
                @endif
            @endforeach
        </div>
        <div class="header-content">
            <div class="title">
                <h1>{{ $content['title'] }}</h1>
                @isset($content['subTitle'])
                    <div class="subtitle">
                        {!! $content['subTitle'] !!}
                    </div>
                @endisset
                <div class="btn-collection">
                    @php
                        $urls = [
                            'services/cloud-solutions/iaas',
                            'services/cloud-solutions/cloud'
                        ];
                    @endphp
                    @if(count($blocksHelper->getBlocksByType('free-consultation', $fields)) > 0)
                        @if(in_array(Request::path(), $urls))
                            <a id="#free-consultation" class="gradient"
                               href="#free-consultation">{{ __('content.Получить демо-доступ') }}</a>
                        @else
                            <a id="#free-consultation" class="gradient"
                               href="#free-consultation">{{ __('content.Получить консультацию') }}</a>
                        @endif
                    @endif
                    @if(count($blocksHelper->getBlocksByType('calculator', $fields)) > 0)
                        @if(in_array(Request::path(), $urls))
                            <a href="#calculator" class="btn-count-price">{{ __('content.Рассчитать стоимость') }}</a>
                        @else
                            <a href="#calculator" class="btn-count-price">{{ __('content.Демо-доступ') }}</a>
                        @endif
                    @endif
                </div>
            </div>
            <div class="title-cod" style="background-image: url('{{ $content['img'] }}')">
            </div>
        </div>
    </div>
</div>
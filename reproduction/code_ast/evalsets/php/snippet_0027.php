@extends('layouts.home')

@section('title', $data->title . " " )

@section('slider')

@endsection
@section('content')
    <section class="breadcrumbs mt-10" style="margin-top: 70px; ">
        <div class="container">

            <div class="d-flex justify-content-between align-items-center">
                <h2>{{$data->title}}</h2>
                <ol>
                    <li><a href="{{route('home')}}">Anasayfa</a></li>
                    <li><a href="">{{$data->title}}</a></li>
                </ol>
            </div>
        </div>
    </section><!-- End Contact Section -->

    <section class="blog">
        <div class="container">

            <div class="row">

                <div class="col-lg-8 entries">

                    @foreach($datalist as $rs)
                        <article class="entry">

                            <div class="entry-img">
                                <img src="{{Storage::url($rs->image)}}" alt="" class="img-fluid" style="width: 100%">
                            </div>

                            <h2 class="entry-title">
                                <a href="{{route('contentdetail', ['id' => $rs->id])}}">{{$rs->title}}</a>
                            </h2>
                            @php
                                $avgcomment =  \App\Http\Controllers\HomeController::avgcomment($data->id);
                                $countcomment = \App\Http\Controllers\HomeController::countcomment($data->id);
                            @endphp
                            <div class="entry-meta">
                                <ul style="display: flex;justify-content: space-between;">
                                    <li class="d-flex align-items-center"><i class="icofont-wall-clock"></i> <a href="">
                                            <time datetime="2020-01-01">{{$rs->created_at->format('d/M/Y')}}</time>
                                        </a></li>
                                    <li class="d-flex align-items-center"><i class="icofont-comment"></i> <a
                                            href="">{{$countcomment}} Yorum</a></li>
                                </ul>
                            </div>

                            <div class="entry-content">
                                <p>
                                    {{$rs->detail}}
                                </p>
                                <div class="read-more">
                                    <a href="{{route('contentdetail', ['id' => $rs->id])}}">Devamı</a>
                                </div>
                            </div>

                        </article><!-- End blog entry -->
                    @endforeach


                    <div class="blog-pagination">
                        <ul class="justify-content-center">
                            <li class="disabled"><i class="icofont-rounded-left"></i></li>
                            <li><a href="#">1</a></li>
                            <li class="active"><a href="#">2</a></li>
                            <li><a href="#">3</a></li>
                            <li><a href="#"><i class="icofont-rounded-right"></i></a></li>
                        </ul>
                    </div>

                </div><!-- End blog entries list -->

                <div class="col-lg-4">
                    <div class="sidebar">
                        <div class="row mb-2">
                            <div class="col-md-2">
                                <img src="{{asset('assets')}}/img/takvim.png" alt="">
                            </div>
                            <div class="col-md-9">
                                <h3 class="sidebar-title">Etkinlikler</h3>
                            </div>
                        </div>
                        <hr>


                        <div class="sidebar-item recent-posts">
                            @foreach($activities as $rs)
                                <div class="post-item clearfix">
                                    <img src="{{asset('assets')}}/img/schedule.png" alt="">

                                    <h6><a href="{{route('contentdetail', ['id' => $rs->id])}}">{{$rs->title}}</a></h6>
                                </div>
                            @endforeach
                        </div><!-- End sidebar recent posts-->

                    </div><!-- End sidebar -->

                </div><!-- End blog sidebar -->

            </div><!-- End .row -->

        </div><!-- End .container -->

    </section><!-- End Blog Section -->



@endsection
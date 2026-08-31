//
//  ServerMockCtl.h
//  TreasureChest
//
//  Created by <NAME> on 2020/2/25.
//  Copyright © 2020 <NAME>. All rights reserved.
//

#import "BaseViewController.h"

NS_ASSUME_NONNULL_BEGIN

@interface ServerMockCtl : BaseViewController

@end

NS_ASSUME_NONNULL_END

/**
    参考1：https://www.jianshu.com/p/74dfec71638a
    2020年02月25日14:48:47  最近疫情导致外围无法访问（easy mock 官网）。等待可以再说。
 
    参考2：https://cloud.tencent.com/developer/article/1388973
    通过GitHub搭建mock数据。

    ：https://juejin.im/post/5df6de3751882512227375d1
 
    相关：
     经过分析，我这里先来说说如何使用面向AOP编程的思想来对接口进行mock操作。NSURLProtocol iOS开发的利器，几乎可以拦截应用内所有的网络请求（WKWebview除外）
     NSURLProtocol 可以实现：
     1）重定向网络请求
     2）忽略网络请求，使用本地缓存
     3）自定义网络请求的返回结果
     4）一些全局的网络请求设置

     我们这里主要是使用自定义网络请求的返回结果的功能。

 
 */
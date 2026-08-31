//
//  SNGetApiTool.h
//  SportNews
//
//  Created by kiss on 2021/7/12.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface SNGetApiTool : NSObject

@property (nonatomic, copy) void(^baseURL)(void);

/// 单例
+ (instancetype)sharedInstance;

//获取域名
- (void)getApiMethod;

@end

NS_ASSUME_NONNULL_END
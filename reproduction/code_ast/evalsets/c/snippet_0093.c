//
//  INSUserProfileViewModel.h
//  Bolts
//
//  Created by <NAME> on 2021/7/5.
//

#import <Foundation/Foundation.h>

#import "INSParseUIConstants.h"

NS_ASSUME_NONNULL_BEGIN

@interface INSUserProfileViewModel : NSObject

@property (nonatomic, strong) PFUser *user;

@property (nonatomic, copy) NSString *userName;
@property (nonatomic, assign) NSInteger followCount;
@property (nonatomic, assign) NSInteger followedCount;
@property (nonatomic, copy) NSString *profileImageLink;
@property (nonatomic, copy) NSString *backgroundImageLink;

@property (nonatomic, strong) UIImage *profileImage;
@property (nonatomic, strong) UIImage *backgroundImage;

@property (nonatomic, assign) BOOL isFollowedByCurrentUser;

- (instancetype)init;
- (instancetype)initWithUser:(PFUser *)user withBlock:(INSBooleanResultBlock)block;

- (void)changeFollowedByCurrentUserStatus:(INSBooleanResultBlock)block;

- (void)logout:(void(^)(void))block;
- (void)unsubscribe:(void(^)(void))block;

@end

NS_ASSUME_NONNULL_END
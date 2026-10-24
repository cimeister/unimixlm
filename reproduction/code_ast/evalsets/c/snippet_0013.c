//
//  FLKNavigationBar.h
//  FLKBaseClasses
//
//  Created by nanhujiaju on 2017/4/19.
//  Copyright © 2017年 nanhu. All rights reserved.
//

#import <UIKit/UIKit.h>

/**
 3ks for https://github.com/allenhsu/CRNavigationController
 */
@interface FLKNavigationBar : UINavigationBar

/**
 * Determines whether or not the extra color layer should be displayed.
 * @param display a BOOL; YES for keeping it visible, NO to hide it.
 * @warning this method is not available in the actual implementation, and is only here for demonstration purposes.
 */
- (void)displayColorLayer:(BOOL)display;

@end
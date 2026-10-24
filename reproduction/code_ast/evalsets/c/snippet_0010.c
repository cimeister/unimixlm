//
//  NSManagedObject+HLSValidationTestCase.h
//  CoconutKit-test
//
//  Created by <NAME> on 10.12.11.
//  Copyright (c) 2011 Hortis. All rights reserved.
//

// Forward declarations
@class ConcreteClassD;

@interface NSManagedObject_HLSValidationTestCase : GHTestCase {
@private
    ConcreteClassD *m_lockedDInstance;
}

@end
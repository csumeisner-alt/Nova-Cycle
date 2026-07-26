package com.novacycle.di

import com.novacycle.billing.BackendMintEntitlementVerifier
import com.novacycle.billing.MintEntitlementVerifier
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/** Binds the backend-based purchase verifier used by BillingManager. */
@Module
@InstallIn(SingletonComponent::class)
abstract class BillingModule {

    @Binds
    @Singleton
    abstract fun bindMintEntitlementVerifier(
        impl: BackendMintEntitlementVerifier
    ): MintEntitlementVerifier
}

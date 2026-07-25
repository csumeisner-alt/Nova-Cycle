package com.novacycle.ui.components

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.ExperimentalMaterialApi
import androidx.compose.material.pullrefresh.PullRefreshIndicator
import androidx.compose.material.pullrefresh.pullRefresh
import androidx.compose.material.pullrefresh.rememberPullRefreshState
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier

/**
 * Shared pull-to-refresh wrapper used by all data screens.
 *
 * Pull-to-refresh relies on nested-scroll events, which only scrollable content
 * emits. Set [contentIsScrollable] = true when [content] already scrolls
 * (LazyColumn, verticalScroll). Otherwise this wrapper adds an invisible
 * full-height scroll container so the pull gesture works on static layouts
 * (chart screens) without changing their layout: the inner box is pinned to
 * the viewport height, so weighted children keep working and nothing actually
 * scrolls.
 *
 * [refreshing] should come from the ViewModel's existing isLoading flag so a
 * successful pull reuses the existing load function and resets the
 * "Updated X ago" label.
 */
@OptIn(ExperimentalMaterialApi::class)
@Composable
fun PullRefreshBox(
    refreshing: Boolean,
    onRefresh: () -> Unit,
    modifier: Modifier = Modifier,
    contentIsScrollable: Boolean = false,
    contentAlignment: Alignment = Alignment.TopStart,
    content: @Composable BoxScope.() -> Unit
) {
    val pullRefreshState = rememberPullRefreshState(refreshing = refreshing, onRefresh = onRefresh)

    Box(modifier = modifier.pullRefresh(pullRefreshState)) {
        if (contentIsScrollable) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = contentAlignment) { content() }
        } else {
            BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
                val viewportHeight = maxHeight
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState())
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(viewportHeight),
                        contentAlignment = contentAlignment
                    ) {
                        content()
                    }
                }
            }
        }

        PullRefreshIndicator(
            refreshing = refreshing,
            state = pullRefreshState,
            modifier = Modifier.align(Alignment.TopCenter),
            backgroundColor = MaterialTheme.colorScheme.surfaceVariant,
            contentColor = MaterialTheme.colorScheme.primary
        )
    }
}

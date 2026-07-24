package com.novacycle.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

/**
 * TickerSelector — placeholder dropdown showing only "VOO".
 *
 * NOTE: Multi-ticker support will be added later.
 * Currently the selector is read-only and always shows "VOO".
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TickerSelector(
    selectedTicker: String = "VOO",
    onTickerSelected: (String) -> Unit = {}
) {
    // Placeholder: only "VOO" is supported. Dropdown is display-only.
    val availableTickers = listOf("VOO")  // Multi-ticker: will be expanded later
    var expanded by remember { mutableStateOf(false) }

    ExposedDropdownMenuBox(
        expanded = false,  // Always closed; interaction disabled until multi-ticker support
        onExpandedChange = { /* No-op: multi-ticker not yet supported */ },
        modifier = Modifier.width(120.dp)
    ) {
        OutlinedTextField(
            value = selectedTicker,
            onValueChange = {},
            readOnly = true,
            label = { Text("Ticker", color = Color(0xFF9E9E9E)) },
            trailingIcon = {
                ExposedDropdownMenuDefaults.TrailingIcon(expanded = false)
            },
            colors = ExposedDropdownMenuDefaults.outlinedTextFieldColors(
                focusedBorderColor = Color(0xFF00C853),
                unfocusedBorderColor = Color(0xFF424242),
                focusedTextColor = Color.White,
                unfocusedTextColor = Color.White
            ),
            modifier = Modifier
                .menuAnchor()
                .fillMaxWidth()
        )

        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            availableTickers.forEach { ticker ->
                DropdownMenuItem(
                    text = { Text(ticker) },
                    onClick = {
                        onTickerSelected(ticker)
                        expanded = false
                    }
                )
            }
        }
    }
}

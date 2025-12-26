import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type Theme = 'dark' | 'light'

interface ThemeState {
    theme: Theme
    setTheme: (theme: Theme) => void
    toggleTheme: () => void
}

export const useThemeStore = create<ThemeState>()(
    persist(
        (set, get) => ({
            theme: 'dark',
            setTheme: (theme) => {
                set({ theme })
                applyTheme(theme)
            },
            toggleTheme: () => {
                const newTheme = get().theme === 'dark' ? 'light' : 'dark'
                set({ theme: newTheme })
                applyTheme(newTheme)
            },
        }),
        {
            name: 'asl-sandbox-theme',
        }
    )
)

function applyTheme(theme: Theme) {
    const root = document.documentElement

    if (theme === 'dark') {
        root.classList.remove('light')
        root.classList.add('dark')
    } else {
        root.classList.remove('dark')
        root.classList.add('light')
    }
}

// Initialize theme on load
export function initializeTheme() {
    const stored = localStorage.getItem('asl-sandbox-theme')
    if (stored) {
        try {
            const { state } = JSON.parse(stored)
            applyTheme(state.theme || 'dark')
        } catch {
            applyTheme('dark')
        }
    } else {
        applyTheme('dark')
    }
}

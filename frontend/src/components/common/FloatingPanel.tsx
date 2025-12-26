import { useState, useRef, useEffect } from 'react'
import { X, Maximize2, Minimize2, Move } from 'lucide-react'

interface FloatingPanelProps {
    title: string
    isOpen: boolean
    onClose: () => void
    children: React.ReactNode
    initialPosition?: { x: number, y: number }
    initialSize?: { width: number, height: number }
}

export default function FloatingPanel({
    title,
    isOpen,
    onClose,
    children,
    initialPosition = { x: 50, y: 50 },
    initialSize = { width: 400, height: 400 }
}: FloatingPanelProps) {
    const [position, setPosition] = useState(initialPosition)
    const [size, setSize] = useState(initialSize)
    const [isDragging, setIsDragging] = useState(false)
    const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
    const [isResizing, setIsResizing] = useState(false)
    const [isMaximized, setIsMaximized] = useState(false)

    // Reset position if window changes size drastically? No, keep it simple.

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (isDragging && !isMaximized) {
                setPosition({
                    x: e.clientX - dragOffset.x,
                    y: e.clientY - dragOffset.y
                })
            }
            if (isResizing && !isMaximized) {
                setSize({
                    width: Math.max(300, e.clientX - position.x),
                    height: Math.max(300, e.clientY - position.y)
                })
            }
        }

        const handleMouseUp = () => {
            setIsDragging(false)
            setIsResizing(false)
        }

        if (isDragging || isResizing) {
            window.addEventListener('mousemove', handleMouseMove)
            window.addEventListener('mouseup', handleMouseUp)
        }
        return () => {
            window.removeEventListener('mousemove', handleMouseMove)
            window.removeEventListener('mouseup', handleMouseUp)
        }
    }, [isDragging, isResizing, dragOffset, position, isMaximized])

    const startDrag = (e: React.MouseEvent) => {
        if (isMaximized) return
        setIsDragging(true)
        setDragOffset({
            x: e.clientX - position.x,
            y: e.clientY - position.y
        })
    }

    const startResize = (e: React.MouseEvent) => {
        e.stopPropagation()
        setIsResizing(true)
    }

    const toggleMaximize = () => {
        setIsMaximized(!isMaximized)
    }

    if (!isOpen) return null

    const style = isMaximized
        ? { top: 0, left: 0, width: '100vw', height: '100vh', zIndex: 50 }
        : { top: position.y, left: position.x, width: size.width, height: size.height, zIndex: 50 }

    return (
        <div
            className="fixed bg-theme-primary border border-theme rounded-lg shadow-2xl flex flex-col overflow-hidden glass"
            style={style}
        >
            {/* Header */}
            <div
                className="h-9 bg-theme-secondary border-b border-theme flex items-center justify-between px-3 cursor-move select-none"
                onMouseDown={startDrag}
            >
                <div className="flex items-center gap-2 text-xs font-semibold text-theme-header">
                    <Move className="w-3 h-3 text-theme-muted" />
                    {title}
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={toggleMaximize} className="p-1 hover:bg-theme-tertiary rounded">
                        {isMaximized ? <Minimize2 className="w-3.5 h-3.5 text-theme-muted" /> : <Maximize2 className="w-3.5 h-3.5 text-theme-muted" />}
                    </button>
                    <button onClick={onClose} className="p-1 hover:bg-red-500/20 hover:text-red-400 rounded transition-colors">
                        <X className="w-3.5 h-3.5 text-theme-muted" />
                    </button>
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 relative overflow-hidden bg-theme-primary">
                {children}
            </div>

            {/* Resize Handle */}
            {!isMaximized && (
                <div
                    className="absolute bottom-0 right-0 w-4 h-4 cursor-nwse-resize flex items-end justify-end p-0.5"
                    onMouseDown={startResize}
                >
                    <div className="w-2 h-2 border-r-2 border-b-2 border-theme-muted opacity-50"></div>
                </div>
            )}
        </div>
    )
}

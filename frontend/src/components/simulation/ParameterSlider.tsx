interface ParameterSliderProps {
    label: string
    value: number
    min: number
    max: number
    step: number
    unit?: string
    onChange: (value: number) => void
}

export default function ParameterSlider({
    label,
    value,
    min,
    max,
    step,
    unit = '',
    onChange,
}: ParameterSliderProps) {
    const percentage = ((value - min) / (max - min)) * 100

    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between">
                <label className="text-sm text-theme-secondary">{label}</label>
                <span className="text-sm font-mono" style={{ color: 'var(--accent-blue)' }}>
                    {value.toFixed(step < 1 ? (step < 0.01 ? 3 : 2) : 0)}{unit}
                </span>
            </div>
            <div className="relative">
                <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={(e) => onChange(parseFloat(e.target.value))}
                    className="w-full h-2 rounded-lg appearance-none cursor-pointer slider"
                    style={{
                        background: `linear-gradient(to right, var(--accent-blue) 0%, var(--accent-blue) ${percentage}%, var(--bg-elevated) ${percentage}%, var(--bg-elevated) 100%)`
                    }}
                />
            </div>
        </div>
    )
}

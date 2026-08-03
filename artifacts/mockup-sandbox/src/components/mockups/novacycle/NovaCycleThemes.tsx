import React, { useState, useRef, MouseEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Activity, Diamond, Settings } from "lucide-react";

// Theme Data
type ThemeId = "DarkLuxe" | "MintLuxe" | "AuroraFlux" | "CrimsonPulse";
interface ThemeDef {
  id: ThemeId;
  name: string;
  unlockTaps: number;
  primary: string;
  bg: string;
}

const THEMES: Record<ThemeId, ThemeDef> = {
  DarkLuxe: { id: "DarkLuxe", name: "DarkLuxe", unlockTaps: 0, primary: "#CBA135", bg: "#0D0D0D" },
  MintLuxe: { id: "MintLuxe", name: "MintLuxe", unlockTaps: 0, primary: "#A8F5D1", bg: "#0D0D0D" },
  AuroraFlux: { id: "AuroraFlux", name: "AuroraFlux", unlockTaps: 10000, primary: "#00D4FF", bg: "#020B24" },
  CrimsonPulse: { id: "CrimsonPulse", name: "CrimsonPulse", unlockTaps: 20000, primary: "#FF0033", bg: "#000000" },
};

type Direction = "observatory" | "terminal" | "minimal";

export function NovaCycleThemes() {
  const [direction, setDirection] = useState<Direction>("observatory");
  const [taps, setTaps] = useState(0);
  const [selectedTheme, setSelectedTheme] = useState<ThemeId>("DarkLuxe");

  const handleSimulate10k = () => setTaps((t) => Math.max(t, 10000));
  const handleSimulate20k = () => setTaps((t) => Math.max(t, 20000));
  const handleReset = () => {
    setTaps(0);
    setSelectedTheme("DarkLuxe");
  };

  const handleSelectTheme = (id: ThemeId) => {
    if (taps >= THEMES[id].unlockTaps) {
      setSelectedTheme(id);
    }
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-white flex flex-col lg:flex-row items-center justify-center p-4 md:p-8 font-sans gap-8 lg:gap-16">
      {/* Controls */}
      <div className="w-full max-w-sm space-y-8 order-2 lg:order-1">
        <div>
          <h2 className="text-2xl font-light mb-6 tracking-wide text-neutral-300">Design Direction</h2>
          <div className="flex flex-col gap-3">
            <DirectionBtn
              active={direction === "observatory"}
              onClick={() => setDirection("observatory")}
              icon={<Sparkles size={18} />}
              title="Observatory"
              desc="Horological luxury, brushed metals"
            />
            <DirectionBtn
              active={direction === "terminal"}
              onClick={() => setDirection("terminal")}
              icon={<Activity size={18} />}
              title="Trading Terminal"
              desc="High-performance, sharp, neon"
            />
            <DirectionBtn
              active={direction === "minimal"}
              onClick={() => setDirection("minimal")}
              icon={<Diamond size={18} />}
              title="Minimal Luxury"
              desc="Editorial jewelry-box, serif"
            />
          </div>
        </div>

        <div>
          <h2 className="text-sm font-semibold mb-3 uppercase tracking-widest text-neutral-500">Simulation Controls</h2>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={handleSimulate10k}
              className="px-4 py-2.5 bg-neutral-900 hover:bg-neutral-800 rounded-md text-sm font-medium border border-neutral-800 transition-colors"
            >
              + 10k Taps
            </button>
            <button
              onClick={handleSimulate20k}
              className="px-4 py-2.5 bg-neutral-900 hover:bg-neutral-800 rounded-md text-sm font-medium border border-neutral-800 transition-colors"
            >
              + 20k Taps
            </button>
            <button
              onClick={handleReset}
              className="col-span-2 px-4 py-2.5 bg-neutral-900/50 hover:bg-neutral-800/50 rounded-md text-sm font-medium border border-neutral-800/50 text-neutral-400 transition-colors"
            >
              Reset Progress
            </button>
          </div>
          <p className="mt-4 text-xs text-neutral-500 leading-relaxed max-w-xs">
            DarkLuxe & MintLuxe are unlocked by default. Aurora unlocks at 10k, Crimson at 20k.
          </p>
        </div>
      </div>

      {/* Phone Frame */}
      <div className="relative order-1 lg:order-2 shrink-0 shadow-2xl shadow-black/80">
        {/* Hardware Frame */}
        <div className="w-[375px] h-[812px] bg-[#111] rounded-[52px] p-3 ring-1 ring-white/10 shadow-[inset_0_0_0_2px_#222,0_0_0_8px_#000] relative overflow-hidden">
          {/* Screen Content */}
          <div
            className="w-full h-full rounded-[40px] overflow-hidden relative"
            style={{ backgroundColor: THEMES[selectedTheme].bg, transition: "background-color 0.5s ease" }}
          >
            {direction === "observatory" && (
              <ObservatoryUI taps={taps} setTaps={setTaps} selectedTheme={selectedTheme} onSelectTheme={handleSelectTheme} />
            )}
            {direction === "terminal" && (
              <TerminalUI taps={taps} setTaps={setTaps} selectedTheme={selectedTheme} onSelectTheme={handleSelectTheme} />
            )}
            {direction === "minimal" && (
              <MinimalUI taps={taps} setTaps={setTaps} selectedTheme={selectedTheme} onSelectTheme={handleSelectTheme} />
            )}
          </div>
          
          {/* Notch mock */}
          <div className="absolute top-3 left-1/2 -translate-x-1/2 w-[140px] h-[32px] bg-black rounded-b-[20px] z-50"></div>
        </div>
      </div>
    </div>
  );
}

function DirectionBtn({ active, onClick, icon, title, desc }: any) {
  return (
    <button
      onClick={onClick}
      className={`text-left p-4 rounded-xl border transition-all duration-300 flex items-start gap-4 ${
        active
          ? "bg-neutral-900 border-neutral-700 shadow-inner"
          : "bg-transparent border-transparent hover:bg-neutral-900/50 hover:border-neutral-800"
      }`}
    >
      <div className={`mt-0.5 p-2 rounded-lg ${active ? "bg-neutral-800 text-white" : "bg-neutral-900 text-neutral-400"}`}>
        {icon}
      </div>
      <div>
        <div className={`font-medium ${active ? "text-white" : "text-neutral-300"}`}>{title}</div>
        <div className="text-xs text-neutral-500 mt-1">{desc}</div>
      </div>
    </button>
  );
}

// -------------------------------------------------------------------------------------------------
// A. Observatory Direction
// -------------------------------------------------------------------------------------------------
function ObservatoryUI({ taps, setTaps, selectedTheme, onSelectTheme }: any) {
  const currentTheme = THEMES[selectedTheme as ThemeId];
  const [particles, setParticles] = useState<{ id: number; x: number; y: number }[]>([]);
  const particleId = useRef(0);

  const handleTap = (e: MouseEvent) => {
    setTaps((t: number) => t + 1);
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const id = particleId.current++;
    setParticles((p) => [...p, { id, x, y }]);
    setTimeout(() => {
      setParticles((p) => p.filter((pt) => pt.id !== id));
    }, 600);
  };

  return (
    <div className="flex flex-col h-full relative" style={{ fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif" }}>
      {/* Background glass gradient effect */}
      <div className="absolute inset-0 pointer-events-none opacity-20"
        style={{
          background: `radial-gradient(circle at 50% 30%, ${currentTheme.primary}40 0%, transparent 60%)`
        }}
      />
      
      {/* Top Bar */}
      <div className="flex justify-between items-center px-6 pt-12 pb-4 z-10">
        <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center backdrop-blur-md border border-white/10">
          <Settings size={16} className="text-white/70" />
        </div>
        
        {/* Observatory Counter: small, elegant number that increments with a subtle scale pop */}
        <motion.div 
          key={taps}
          initial={{ scale: 1.2, color: currentTheme.primary }}
          animate={{ scale: 1, color: "#ffffff" }}
          transition={{ duration: 0.3 }}
          className="text-lg tracking-widest font-light tabular-nums"
        >
          {taps.toLocaleString()}
        </motion.div>
      </div>

      {/* Central Tap Zone */}
      <div className="flex-1 flex flex-col items-center justify-center z-10 relative px-8">
        <motion.div
          whileTap={{ scale: 0.95 }}
          onClick={handleTap}
          className="relative w-56 h-56 rounded-full flex items-center justify-center cursor-pointer select-none"
        >
          {/* Metallic rim */}
          <div className="absolute inset-0 rounded-full border border-white/10 shadow-[inset_0_0_20px_rgba(255,255,255,0.05)] bg-gradient-to-br from-white/5 to-transparent" />
          
          <div className="text-3xl tracking-[0.2em] font-light" style={{ color: currentTheme.primary }}>
            NOVA
          </div>

          {/* Particles */}
          <AnimatePresence>
            {particles.map((p) => (
              <motion.div
                key={p.id}
                initial={{ opacity: 1, scale: 0, x: p.x - 28, y: p.y - 28 }}
                animate={{ 
                  opacity: 0, 
                  scale: Math.random() * 2 + 1,
                  x: p.x - 28 + (Math.random() - 0.5) * 120,
                  y: p.y - 28 + (Math.random() - 0.5) * 120 
                }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.6, ease: "easeOut" }}
                className="absolute w-1.5 h-1.5 rounded-full blur-[1px]"
                style={{ backgroundColor: currentTheme.primary, boxShadow: `0 0 10px ${currentTheme.primary}`, pointerEvents: "none" }}
              />
            ))}
          </AnimatePresence>
        </motion.div>
        
        <div className="mt-12 text-center">
          <div className="text-white/40 text-[10px] tracking-[0.3em] uppercase mb-1">Status</div>
          <div className="text-white/80 text-sm font-light tracking-wide">Monitoring</div>
        </div>
      </div>

      {/* Theme Picker: large circular swatches, metallic rim, soft glow. Locked = faint progress arc */}
      <div className="pb-10 pt-8 px-6 bg-gradient-to-t from-black/90 via-black/60 to-transparent z-10 border-t border-white/5 backdrop-blur-sm relative">
        <div className="flex justify-between items-center gap-3">
          {(Object.values(THEMES) as ThemeDef[]).map((theme) => {
            const isUnlocked = taps >= theme.unlockTaps;
            const isSelected = selectedTheme === theme.id;
            const progress = theme.unlockTaps > 0 ? Math.min(1, taps / theme.unlockTaps) : 1;
            
            return (
              <div key={theme.id} className="flex flex-col items-center gap-3">
                <button
                  onClick={() => onSelectTheme(theme.id)}
                  disabled={!isUnlocked}
                  className="relative w-14 h-14 rounded-full flex items-center justify-center group"
                >
                  {/* Base Circle */}
                  <div 
                    className="absolute inset-2.5 rounded-full transition-all duration-300"
                    style={{ 
                      backgroundColor: isUnlocked ? theme.primary : "#1A1A1A",
                      boxShadow: isSelected ? `0 0 20px ${theme.primary}60` : 'none',
                      opacity: isUnlocked ? 1 : 0.4
                    }} 
                  />
                  
                  {/* Metallic Rim / Ring */}
                  <div className="absolute inset-0 rounded-full border border-white/10 bg-gradient-to-br from-white/5 to-transparent" />
                  
                  {/* Progress Arc for locked */}
                  {!isUnlocked && (
                    <svg className="absolute inset-0 w-full h-full -rotate-90" viewBox="0 0 100 100">
                      <circle 
                        cx="50" cy="50" r="48" 
                        fill="transparent" 
                        stroke={theme.primary} 
                        strokeWidth="2" 
                        strokeDasharray="301.59" 
                        strokeDashoffset={301.59 * (1 - progress)} 
                        strokeLinecap="round"
                        className="opacity-60 transition-all duration-500"
                      />
                    </svg>
                  )}
                  
                  {/* Selected Indicator */}
                  {isSelected && (
                    <div className="absolute inset-[-4px] border-[1px] border-white/20 rounded-full transition-transform" />
                  )}
                </button>
                <span className="text-[10px] text-white/50 tracking-wider">
                  {theme.name.replace('Luxe','').replace('Flux','').replace('Pulse','')}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------------------------------------------
// B. Trading Terminal Direction
// -------------------------------------------------------------------------------------------------
function TerminalUI({ taps, setTaps, selectedTheme, onSelectTheme }: any) {
  const currentTheme = THEMES[selectedTheme as ThemeId];
  const [ripples, setRipples] = useState<{ id: number; x: number; y: number }[]>([]);
  const [floaters, setFloaters] = useState<{ id: number; x: number; y: number }[]>([]);
  const idCounter = useRef(0);

  const handleTap = (e: MouseEvent) => {
    setTaps((t: number) => t + 1);
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // offset so the floating +1 goes outward
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const offsetX = (x - centerX) * 1.5;
    const offsetY = (y - centerY) * 1.5;
    
    const id = idCounter.current++;
    setRipples((r) => [...r, { id, x, y }]);
    setFloaters((f) => [...f, { id, x: centerX + offsetX, y: centerY + offsetY }]);
    
    setTimeout(() => {
      setRipples((r) => r.filter((item) => item.id !== id));
      setFloaters((f) => f.filter((item) => item.id !== id));
    }, 800);
  };

  return (
    <div className="flex flex-col h-full relative bg-[#050505]" style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}>
      {/* Background grid */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.03] bg-[linear-gradient(to_right,#ffffff_1px,transparent_1px),linear-gradient(to_bottom,#ffffff_1px,transparent_1px)] bg-[size:24px_24px]" />
      
      {/* Top Bar: Bold monospace digits like a ticker */}
      <div className="flex justify-between items-center px-5 pt-12 pb-4 border-b border-white/10 z-10 bg-[#050505]/90 backdrop-blur-md">
        <div className="text-[10px] text-white/40 uppercase tracking-widest">SYS_RDY</div>
        <div className="flex items-center gap-3">
          <div className="text-[10px] text-white/30">VOL</div>
          <div className="text-xl font-bold tracking-tight tabular-nums" style={{ color: currentTheme.primary, textShadow: `0 0 10px ${currentTheme.primary}40` }}>
            {taps.toLocaleString().padStart(6, '0')}
          </div>
        </div>
      </div>

      {/* Central Tap Zone */}
      <div className="flex-1 flex flex-col items-center justify-center z-10 relative overflow-hidden">
        {/* Concentric scan lines */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-20">
          <div className="w-64 h-64 rounded-full border border-white/20 border-dashed" />
          <div className="absolute w-48 h-48 rounded-full border border-white/10" />
          <div className="absolute w-32 h-32 rounded-full border border-white/5" />
          <div className="absolute w-full h-[1px] bg-white/5" />
          <div className="absolute h-full w-[1px] bg-white/5" />
        </div>

        <motion.div
          whileTap={{ scale: 0.96 }}
          onClick={handleTap}
          className="relative w-48 h-48 flex items-center justify-center cursor-pointer select-none"
        >
          {/* Target box */}
          <div className="absolute inset-0 border-2 border-transparent opacity-60">
            <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-white/60" />
            <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-white/60" />
            <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-white/60" />
            <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-white/60" />
          </div>
          
          <div className="text-3xl font-bold tracking-[0.2em] ml-1" style={{ color: currentTheme.primary }}>
            N/C
          </div>

          {/* Ripples */}
          <AnimatePresence>
            {ripples.map((r) => (
              <motion.div
                key={r.id}
                initial={{ opacity: 0.8, scale: 0, x: r.x - 96, y: r.y - 96 }}
                animate={{ opacity: 0, scale: 2.5 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5, ease: "easeOut" }}
                className="absolute w-[192px] h-[192px] border border-white rounded-full pointer-events-none"
                style={{ borderColor: currentTheme.primary, boxShadow: `0 0 15px ${currentTheme.primary}` }}
              />
            ))}
          </AnimatePresence>

          {/* Floaters (+1) */}
          <AnimatePresence>
            {floaters.map((f) => (
              <motion.div
                key={f.id}
                initial={{ opacity: 1, y: f.y - 96, x: f.x - 96, scale: 0.5 }}
                animate={{ opacity: 0, y: f.y - 150, scale: 1.2 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.6, ease: "easeOut" }}
                className="absolute text-base font-bold pointer-events-none"
                style={{ color: currentTheme.primary, textShadow: `0 0 5px ${currentTheme.primary}` }}
              >
                +1
              </motion.div>
            ))}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* Theme Picker: Segmented tab-like swatches, glowing underline */}
      <div className="px-4 pb-8 pt-4 bg-[#050505] border-t border-white/10 z-10">
        <div className="flex w-full bg-white/5 rounded-md p-1">
          {(Object.values(THEMES) as ThemeDef[]).map((theme) => {
            const isUnlocked = taps >= theme.unlockTaps;
            const isSelected = selectedTheme === theme.id;
            
            return (
              <button
                key={theme.id}
                onClick={() => onSelectTheme(theme.id)}
                disabled={!isUnlocked}
                className="flex-1 relative py-3.5 text-[10px] font-bold tracking-widest uppercase overflow-hidden"
              >
                <span 
                  className="relative z-10 transition-colors"
                  style={{ color: isSelected ? theme.primary : isUnlocked ? "rgba(255,255,255,0.7)" : "rgba(255,255,255,0.15)" }}
                >
                  {theme.name.slice(0, 3)}
                </span>
                
                {/* Selection underline / glow */}
                {isSelected && (
                  <motion.div 
                    layoutId="terminalTab"
                    className="absolute bottom-0 left-0 w-full h-[2px] z-0"
                    style={{ backgroundColor: theme.primary, boxShadow: `0 -2px 10px ${theme.primary}` }}
                  />
                )}
                
                {/* Progress bar background for locked */}
                {!isUnlocked && (
                  <div 
                    className="absolute bottom-0 left-0 h-[2px] bg-white/20 z-0 transition-all duration-500"
                    style={{ width: `${Math.min(100, (taps / theme.unlockTaps) * 100)}%` }}
                  />
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------------------------------------------
// C. Minimal Luxury Direction
// -------------------------------------------------------------------------------------------------
function MinimalUI({ taps, setTaps, selectedTheme, onSelectTheme }: any) {
  const currentTheme = THEMES[selectedTheme as ThemeId];
  const [gems, setGems] = useState<{ id: number; x: number; y: number }[]>([]);
  const idCounter = useRef(0);
  
  // Custom theme colors for the minimalist background
  const bgMap: Record<ThemeId, string> = {
    DarkLuxe: "#100f0d",
    MintLuxe: "#0c110e",
    AuroraFlux: "#0a0c12",
    CrimsonPulse: "#120a0b",
  };

  const handleTap = (e: MouseEvent) => {
    setTaps((t: number) => t + 1);
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const id = idCounter.current++;
    setGems((g) => [...g, { id, x, y }]);
    
    setTimeout(() => {
      setGems((g) => g.filter((item) => item.id !== id));
    }, 1200);
  };

  return (
    <div 
      className="flex flex-col h-full relative transition-colors duration-[1500ms]" 
      style={{ 
        backgroundColor: bgMap[selectedTheme as ThemeId],
      }}
    >
      {/* Apply standard font to whole container, use playfair just for headings/brand */}
      
      {/* Top Bar: subtle floating badge near the logo... but header is useful. */}
      <div className="flex justify-between items-center px-6 pt-12 pb-4 z-10 font-sans">
        <div className="text-[9px] tracking-[0.2em] text-white/40">SETTINGS</div>
        <div className="text-[9px] tracking-[0.2em] text-white/40">MENU</div>
      </div>

      {/* Central Tap Zone */}
      <div className="flex-1 flex flex-col items-center justify-center z-10 relative">
        <motion.div
          whileTap={{ scale: 0.98 }}
          onClick={handleTap}
          className="relative flex flex-col items-center justify-center cursor-pointer select-none w-full h-64"
        >
          {/* Logo pulses briefly on tap */}
          <motion.div 
            className="text-5xl tracking-[0.3em] font-light mb-8 relative z-10"
            style={{ 
              color: "#ffffff",
              fontFamily: "'Playfair Display', ui-serif, Georgia, Cambria, 'Times New Roman', Times, serif" 
            }}
            whileTap={{ color: currentTheme.primary, textShadow: `0 0 30px ${currentTheme.primary}40`, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            NOVA
          </motion.div>

          {/* Floating badge for counter near the logo */}
          <div className="flex items-center gap-2.5 px-4 py-1.5 rounded-full border border-white/[0.08] bg-white/[0.02] backdrop-blur-md">
            <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: currentTheme.primary }} />
            <div className="text-[11px] font-sans tracking-widest text-white/70 tabular-nums">
              {taps.toLocaleString()}
            </div>
          </div>

          {/* Gem particles */}
          <AnimatePresence>
            {gems.map((g) => (
              <motion.div
                key={g.id}
                initial={{ opacity: 0, y: 0, x: g.x - (typeof window !== 'undefined' ? window.innerWidth / 2 : 180), scale: 0 }}
                animate={{ opacity: [0, 1, 0], y: -120 - Math.random() * 60, x: g.x - 180 + (Math.random() - 0.5) * 60, scale: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 1.2, ease: [0.25, 0.1, 0.25, 1] }}
                className="absolute w-1.5 h-1.5 rotate-45 pointer-events-none z-0"
                style={{ backgroundColor: currentTheme.primary, boxShadow: `0 0 10px ${currentTheme.primary}` }}
              />
            ))}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* Theme Picker: Vertical list of "gemstone" chips */}
      <div className="px-6 pb-12 pt-4 z-10 flex flex-col gap-2">
        <div className="text-[10px] tracking-[0.2em] text-white/30 mb-2 ml-1 font-sans">THEMES</div>
        {(Object.values(THEMES) as ThemeDef[]).map((theme) => {
          const isUnlocked = taps >= theme.unlockTaps;
          const isSelected = selectedTheme === theme.id;
          const progress = theme.unlockTaps > 0 ? taps / theme.unlockTaps : 1;
          
          return (
            <button
              key={theme.id}
              onClick={() => onSelectTheme(theme.id)}
              disabled={!isUnlocked}
              className="relative w-full flex items-center justify-between p-4 rounded-xl transition-all duration-500 overflow-hidden group border"
              style={{
                backgroundColor: isSelected ? 'rgba(255,255,255,0.03)' : 'transparent',
                borderColor: isSelected ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.02)'
              }}
            >
              <div className="flex items-center gap-5 relative z-10">
                <div 
                  className="w-1 h-6 rounded-full transition-colors duration-500"
                  style={{ backgroundColor: isUnlocked ? theme.primary : 'rgba(255,255,255,0.1)' }}
                />
                <div className="flex flex-col items-start font-sans">
                  <span className={`text-[13px] tracking-[0.15em] font-light transition-colors duration-500 ${isUnlocked ? 'text-white/90' : 'text-white/30'}`}>
                    {theme.name.toUpperCase()}
                  </span>
                  {!isUnlocked && (
                    <span className="text-[9px] tracking-wider text-white/30 mt-1 uppercase">
                      {theme.unlockTaps.toLocaleString()} Taps
                    </span>
                  )}
                </div>
              </div>
              
              {/* Progress background for locked item */}
              {!isUnlocked && (
                <div className="absolute top-0 bottom-0 left-0 bg-white/[0.02] z-0 transition-all duration-500"
                     style={{ width: `${Math.min(100, progress * 100)}%` }}
                />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

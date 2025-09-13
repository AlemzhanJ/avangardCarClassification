'use client';

import Image from 'next/image';
import { useEffect, useState } from 'react';

interface LogoProps {
  className?: string;
  width?: number;
  height?: number;
}

export default function Logo({ className = '', width = 120, height = 40 }: LogoProps) {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const checkDarkMode = () => {
      setIsDark(window.matchMedia('(prefers-color-scheme: dark)').matches);
    };

    checkDarkMode();
    
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    mediaQuery.addListener(checkDarkMode);

    return () => mediaQuery.removeListener(checkDarkMode);
  }, []);

  return (
    <Image
      src={isDark ? "/logo.svg" : "/logo-dark.svg"}
      alt="inDrive Logo"
      width={width}
      height={height}
      className={className}
      priority
    />
  );
}
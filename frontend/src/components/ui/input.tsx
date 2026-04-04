import * as React from "react";
import { cn } from "@/lib/utils";

const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(
  ({ className, type, ...props }, ref) => (
    <input
      type={type}
      ref={ref}
      className={cn(
        "flex h-11 w-full rounded-xl border border-border/80 bg-white/4 px-4 py-2 text-sm text-foreground shadow-inner shadow-black/20 transition-colors placeholder:text-muted-foreground/80 focus-visible:ring-2 focus-visible:ring-ring/80 focus-visible:outline-none",
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";

export { Input };

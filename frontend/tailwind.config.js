module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        indigo: {
          50: '#eef2ff',
          100: '#e0e7ff',
          600: '#4f46e5',
          700: '#4338ca',
          900: '#312e81',
        },
      },
      animation: {
        'fade-in-down': 'fadeIn 1s ease-out',
        'fade-in-up': 'fadeInUp 0.5s ease-out forwards',
        'blob': 'blob 7s infinite',
        'blob-x': 'blobX 10s infinite',
        'blob-y': 'blobY 12s infinite',
        'blob-xy': 'blobXY 14s infinite',
        'blob-reverse': 'blobReverse 11s infinite',
        'scale': 'scale 0.5s ease-out',
        'scale-up': 'scaleUp 0.3s ease-out forwards',
        'slide-up': 'slideUp 0.5s ease-out forwards',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(-10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scale: {
          '0%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.05)' },
          '100%': { transform: 'scale(1)' },
        },
        scaleUp: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        blob: {
          '0%': {
            transform: 'translate(0px, 0px) scale(1)',
          },
          '33%': {
            transform: 'translate(30px, -50px) scale(1.1)',
          },
          '66%': {
            transform: 'translate(-20px, 20px) scale(0.9)',
          },
          '100%': {
            transform: 'translate(0px, 0px) scale(1)',
          },
        },
        blobX: {
          '0%': { transform: 'translateX(0px) scale(1)' },
          '50%': { transform: 'translateX(300px) scale(1.1)' },
          '100%': { transform: 'translateX(0px) scale(1)' },
        },
        blobY: {
          '0%': { transform: 'translateY(0px) scale(1)' },
          '50%': { transform: 'translateY(200px) scale(1.1)' },
          '100%': { transform: 'translateY(0px) scale(1)' },
        },
        blobXY: {
          '0%': { transform: 'translate(0px, 0px) scale(1)' },
          '25%': { transform: 'translate(150px, 100px) scale(1.1)' },
          '75%': { transform: 'translate(-150px, -100px) scale(0.9)' },
          '100%': { transform: 'translate(0px, 0px) scale(1)' },
        },
        blobReverse: {
          '0%': { transform: 'translateX(0px) scale(1)' },
          '50%': { transform: 'translateX(-300px) scale(1.1)' },
          '100%': { transform: 'translateX(0px) scale(1)' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}

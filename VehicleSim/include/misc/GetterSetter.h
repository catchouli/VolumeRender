#ifndef VEHICLESIM_GETTERSETTER
#define VEHICLESIM_GETTERSETTER

namespace vlr
{
	template <typename baseClass, typename pointerType>
	class GetterSetter
	{
	private:
		enum Mode;

	public:
		typedef pointerType (baseClass::*getterPointer)() const;
		typedef const pointerType& (baseClass::*getterPointerConstRef)() const;
		typedef void (baseClass::*setterPointer)(pointerType);
		typedef void (baseClass::*setterPointerConstRef)(const pointerType&);
		typedef pointerType baseClass::*finalPointerType;

		GetterSetter(baseClass* base, finalPointerType pointer)
			: _mode(POINTER), _pointer(pointer), _base(base)
		{

		}

		GetterSetter(baseClass* base, getterPointer getterFunc)
			: _mode(FUNC), _getterFunc(getterFunc), _setterFunc(nullptr),
			_base(base), _setterFuncConstRef(nullptr)
		{

		}

		GetterSetter(baseClass* base, getterPointer getterFunc,
			setterPointer setterFunc)
			: _mode(FUNC), _getterFunc(getterFunc), _setterFunc(setterFunc),
			_base(base)
		{

		}

		GetterSetter(baseClass* base, getterPointer getterFunc,
			setterPointerConstRef setterFuncConstRef)
			: _mode(FUNC_SETCONSTREF), _getterFunc(getterFunc),
			_setterFuncConstRef(setterFuncConstRef),
			_base(base)
		{

		}

		GetterSetter(baseClass* base, getterPointerConstRef getterFunc)
			: _mode(FUNC_CONSTREF), _getterFuncConstRef(getterFunc),
			_setterFuncConstRef(nullptr), _base(base),
			_setterFunc(nullptr)
		{

		}

		GetterSetter(baseClass* base, getterPointerConstRef getterFunc,
			setterPointerConstRef setterFuncConstRef)
			: _mode(FUNC_CONSTREF), _getterFuncConstRef(getterFunc),
			_setterFuncConstRef(setterFuncConstRef), _base(base)
		{

		}

		pointerType getValue()
		{
			if (_base == nullptr)
				return pointerType();

			switch (_mode)
			{
			case POINTER:
				if (_pointer != nullptr)
					return (_base->*_pointer);
				break;
			case FUNC:
			case FUNC_SETCONSTREF:
				if (_getterFunc != nullptr)
					return (_base->*_getterFunc)();
				break;
			case FUNC_CONSTREF:
				if (_getterFuncConstRef != nullptr)
					return (_base->*_getterFuncConstRef)();
				break;
			}

			return pointerType();
		}

		void setBase(baseClass* base)
		{
			_base = base;
		}

		baseClass* getBase() const
		{
			return _base;
		}

		void setValue(pointerType val)
		{
			if (_base == nullptr)
				return;

			switch (_mode)
			{
			case POINTER:
				if (_pointer != nullptr)
					(_base->*_pointer) = val;
				break;
			case FUNC:
				if (_setterFunc != nullptr)
					(_base->*_setterFunc)(val);
				break;
			case FUNC_CONSTREF:
			case FUNC_SETCONSTREF:
				if (_setterFuncConstRef != nullptr)
					(_base->*_setterFuncConstRef)(val);
				break;
			default:
				break;
			}
		}
		
	private:
		enum Mode
		{
			POINTER,
			FUNC,
			FUNC_SETCONSTREF,
			FUNC_CONSTREF
		};

		Mode _mode;

		baseClass* _base;
		finalPointerType _pointer;
		getterPointer _getterFunc;
		getterPointerConstRef _getterFuncConstRef;
		setterPointer _setterFunc;
		setterPointerConstRef _setterFuncConstRef;
	};
}

#endif /* VEHICLESIM_GETTERSETTER */
